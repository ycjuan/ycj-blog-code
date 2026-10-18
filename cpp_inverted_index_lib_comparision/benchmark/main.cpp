// Benchmark: generates a synthetic corpus, builds both the PISA-style and
// Xapian inverted-index engines, runs an identical set of boolean queries
// (from simple single-term to multi-level nested AND/OR/NOT) against both,
// and reports result count + latency for each engine/query pair. Also
// cross-checks that both engines agree on the result doc-id set, as a
// correctness sanity check.
//
// A second section benchmarks a "phrase-as-key" scenario: a simple
// unordered_map<string, vector<docId>> KeyEngine where a whole multi-term
// key (e.g. "vocA vocB vocC") is the lookup unit, compared against PISA and
// Xapian answering the equivalent classic AND query.
//
// Usage:
//   ./benchmark [--num_docs=N] [--num_vocab_per_doc=N] [--num_char_per_vocab=N]
//               [--vocab_card=N] [--reps=N] [--num_vocab_per_key=N] [--num_keys_per_doc=N]
//
// Build: see Makefile (requires `xapian-config` on PATH).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "corpus_generator.h"
#include "key_engine.h"
#include "pisa_engine.h"
#include "query_ast.h"
#include "xapian_engine.h"

using namespace std;
using Clock = chrono::high_resolution_clock;

struct Args
{
    CorpusConfig corpus;
    int          numVocabPerKey = 3; // terms per key, for the phrase-as-key benchmark
    int          numKeysPerDoc  = 5; // keys per doc, for the phrase-as-key benchmark
    int          reps           = 20; // repetitions per query; reports p50/p95 latency across these reps
};

static void printUsage(const char* program)
{
    Args defaults; // show default values alongside each flag
    cout << "Usage: " << program << " [OPTIONS]\n\n"
         << "Generates a synthetic corpus and benchmarks boolean query latency on a\n"
         << "PISA-style index versus Xapian, cross-checking that both return the same\n"
         << "result doc-id set for every query.\n\n"
         << "Options:\n"
         << "  -d, --num_docs=N            Number of documents to generate (default: " << defaults.corpus.numDocs
         << ")\n"
         << "  -v, --num_vocab_per_doc=N   Terms sampled (with replacement) into each doc (default: "
         << defaults.corpus.numVocabPerDoc << ")\n"
         << "  -l, --num_char_per_vocab=N  Character length of each distinct vocabulary term (default: "
         << defaults.corpus.numCharPerVocab << ")\n"
         << "  -k, --vocab_card=N          Vocabulary cardinality, i.e. # of distinct terms overall (default: "
         << defaults.corpus.vocabCard << ")\n"
         << "  -r, --reps=N                Repetitions per query; reports p50/p95 latency across them (default: "
         << defaults.reps << ")\n"
         << "  -p, --num_vocab_per_key=N   Terms per key, for the phrase-as-key benchmark (default: "
         << defaults.numVocabPerKey << ")\n"
         << "  -n, --num_keys_per_doc=N    Keys per doc, for the phrase-as-key benchmark (default: "
         << defaults.numKeysPerDoc << ")\n"
         << "  -h, --help                  Show this help message and exit\n\n"
         << "Example:\n"
         << "  " << program << " --num_docs=200000 --num_vocab_per_doc=30 --vocab_card=5000 --num_char_per_vocab=8"
         << " --reps=7 --num_vocab_per_key=3 --num_keys_per_doc=5\n"
         << "  " << program << " -d 200000 -v 30 -k 5000 -l 8 -r 7 -p 3 -n 5\n";
}

static Args parseArgs(int argc, char** argv)
{
    Args args;
    // Maps a flag (long or short form) to the Args field it sets.
    auto apply = [&](const string& key, int value)
    {
        if (key == "num_docs" || key == "d")
            args.corpus.numDocs = value;
        else if (key == "num_vocab_per_doc" || key == "v")
            args.corpus.numVocabPerDoc = value;
        else if (key == "num_char_per_vocab" || key == "l")
            args.corpus.numCharPerVocab = value;
        else if (key == "vocab_card" || key == "k")
            args.corpus.vocabCard = value;
        else if (key == "reps" || key == "r")
            args.reps = value;
        else if (key == "num_vocab_per_key" || key == "p")
            args.numVocabPerKey = value;
        else if (key == "num_keys_per_doc" || key == "n")
            args.numKeysPerDoc = value;
    };

    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            exit(0);
        }
        if (arg.rfind("--", 0) == 0)
        {
            // Long form: --key=value
            auto eq = arg.find('=');
            if (eq == string::npos)
                continue;
            apply(arg.substr(2, eq - 2), stoi(arg.substr(eq + 1)));
        }
        else if (arg.rfind("-", 0) == 0 && arg.size() == 2 && i + 1 < argc)
        {
            // Short form: -x value
            apply(arg.substr(1), stoi(argv[++i]));
        }
    }
    return args;
}

// Builds the benchmark query set. Uses the first few vocab terms (guaranteed
// to exist in the corpus) so every query produces a meaningful, non-empty
// result on a non-trivial corpus.
static vector<string> buildQueries(const vector<string>& vocab)
{
    auto v = [&](int i) { return vocab[i]; };
    return {
        v(0),
        v(0) + " AND " + v(1),
        "(" + v(0) + " AND " + v(1) + ") OR " + v(2),
        "(" + v(0) + " AND " + v(1) + ") OR (" + v(2) + " AND " + v(3) + ")",
        v(0) + " AND (" + v(1) + " OR " + v(2) + ") AND NOT " + v(3),
        v(0) + " OR " + v(1) + " OR " + v(2) + " OR " + v(3) + " OR " + v(4),
        "(" + v(0) + " OR " + v(1) + ") AND NOT (" + v(2) + " AND " + v(3) + ")",
    };
}

// Picks a handful of distinct keys (in order of first appearance) from the
// key corpus to use as benchmark queries -- guaranteed to exist, so results
// are non-empty.
static vector<string> buildKeyQueries(const vector<vector<string>>& keysPerDoc, int maxQueries = 5)
{
    vector<string>           queries;
    unordered_set<string> seen;
    for (const auto& keys : keysPerDoc)
    {
        for (const string& key : keys)
        {
            if (seen.insert(key).second)
            {
                queries.push_back(key);
                if (static_cast<int>(queries.size()) >= maxQueries)
                    return queries;
            }
        }
    }
    return queries;
}

// Splits a key string ("vocA vocB vocC") into the equivalent classic boolean
// AND query ("vocA AND vocB AND vocC") that PISA/Xapian can evaluate.
static string keyToAndQuery(const string& key)
{
    istringstream iss(key);
    string        term;
    string        query;
    while (iss >> term)
    {
        if (!query.empty())
            query += " AND ";
        query += term;
    }
    return query;
}

struct LatencyStats
{
    double p50Us = 0;
    double p95Us = 0;
};

// Nearest-rank percentile over a *sorted* ascending vector.
static double percentile(const vector<double>& sortedValues, double p)
{
    size_t n = sortedValues.size();
    size_t rank = static_cast<size_t>(ceil(p / 100.0 * static_cast<double>(n)));
    rank = max<size_t>(rank, 1);
    rank = min(rank, n);
    return sortedValues[rank - 1];
}

// Generic timed evaluator: `evalFn` is any zero-arg callable returning
// vector<int> (e.g. a lambda wrapping engine.evaluate(ast) or
// engine.evaluate(key)), so the same timing/percentile logic works whether
// the engine's query is an AST or a plain string key.
template <typename EvalFn>
static pair<vector<int>, LatencyStats> timedEvaluate(EvalFn&& evalFn, int reps)
{
    vector<double> latenciesUs;
    vector<int>    result;
    for (int r = 0; r < reps; r++)
    {
        auto start = Clock::now();
        result     = evalFn();
        auto end   = Clock::now();
        latenciesUs.push_back(chrono::duration<double, micro>(end - start).count());
    }
    sort(latenciesUs.begin(), latenciesUs.end());
    LatencyStats stats;
    stats.p50Us = percentile(latenciesUs, 50.0);
    stats.p95Us = percentile(latenciesUs, 95.0);
    return { result, stats };
}

int main(int argc, char** argv)
{
    Args                args = parseArgs(argc, argv);
    const CorpusConfig& cfg  = args.corpus;

    cout << "Corpus config: num_docs=" << cfg.numDocs << " num_vocab_per_doc=" << cfg.numVocabPerDoc
         << " num_char_per_vocab=" << cfg.numCharPerVocab << " vocab_card=" << cfg.vocabCard << "\n\n";

    Corpus corpus = generateCorpus(cfg);

    PisaEngine pisaEngine;
    auto       pisaBuildStart = Clock::now();
    pisaEngine.build(corpus.docs);
    double pisaBuildMs = chrono::duration<double, milli>(Clock::now() - pisaBuildStart).count();

    XapianEngine xapianEngine;
    auto         xapianBuildStart = Clock::now();
    xapianEngine.build(corpus.docs);
    double xapianBuildMs = chrono::duration<double, milli>(Clock::now() - xapianBuildStart).count();

    cout << "Index build time: pisa=" << fixed << setprecision(2) << pisaBuildMs << " ms, xapian=" << xapianBuildMs
         << " ms\n";
    cout << "Indexed docs: pisa=" << pisaEngine.numDocs() << ", xapian=" << xapianEngine.numDocs() << "\n\n";

    vector<string> queries = buildQueries(corpus.vocab);

    cout << left << setw(38) << "Query" << right << setw(10) << "PISA p50" << setw(10) << "PISA p95" << setw(11)
         << "PISA QPS" << setw(12) << "Xapian p50" << setw(12) << "Xapian p95" << setw(11) << "Xapian QPS" << setw(9)
         << "PISA#" << setw(9) << "Xapian#" << setw(7) << "Match"
         << "\n";
    cout << left << setw(38) << "" << right << setw(10) << "(ms)" << setw(10) << "(ms)" << setw(11) << "" << setw(12)
         << "(ms)" << setw(12) << "(ms)" << setw(11) << "" << setw(9) << "" << setw(9) << "" << setw(7) << "" << "\n";
    cout << string(139, '-') << "\n";

    for (const string& query : queries)
    {
        unique_ptr<QueryNode> ast = QueryParser::parse(query);

        auto [pisaResult, pisaStats] = timedEvaluate([&] { return pisaEngine.evaluate(*ast); }, args.reps);
        auto [xapianResult, xapianStats] = timedEvaluate([&] { return xapianEngine.evaluate(*ast); }, args.reps);

        // QPS here means single-threaded queries/sec implied by one query's
        // p50 latency (1e6 / p50_latency_us), not a multi-threaded
        // throughput measurement.
        double pisaQps   = 1e6 / pisaStats.p50Us;
        double xapianQps = 1e6 / xapianStats.p50Us;

        bool match = (pisaResult == xapianResult);

        string displayQuery = query.size() > 36 ? query.substr(0, 33) + "..." : query;
        cout << left << setw(38) << displayQuery << right << setw(10) << fixed << setprecision(4)
             << pisaStats.p50Us / 1000.0 << setw(10) << pisaStats.p95Us / 1000.0 << setw(11) << setprecision(0)
             << pisaQps << setw(12) << setprecision(4) << xapianStats.p50Us / 1000.0 << setw(12)
             << xapianStats.p95Us / 1000.0 << setw(11) << setprecision(0) << xapianQps << setw(9)
             << pisaResult.size() << setw(9) << xapianResult.size() << setw(7) << (match ? "yes" : "NO") << "\n";

        if (!match)
        {
            cerr << "  !! Mismatch for query: " << query << "\n";
        }
    }

    // --- Phrase-as-key benchmark ---------------------------------------
    // A separate corpus where each doc is composed of `num_keys_per_doc`
    // keys, each a fixed-size tuple of `num_vocab_per_key` terms. KeyEngine
    // looks the whole key up as one hash-map entry; PISA/Xapian answer the
    // "same" query as a classic boolean AND of the key's individual terms.
    // The two are NOT expected to return identical doc sets: AND matches
    // any doc containing all terms anywhere (even split across different
    // keys), while the key lookup only matches docs having that exact key.
    KeyCorpusConfig keyCfg;
    keyCfg.numDocs         = cfg.numDocs;
    keyCfg.numCharPerVocab = cfg.numCharPerVocab;
    keyCfg.vocabCard       = cfg.vocabCard;
    keyCfg.numVocabPerKey  = args.numVocabPerKey;
    keyCfg.numKeysPerDoc   = args.numKeysPerDoc;

    KeyCorpus keyCorpus = generateKeyCorpus(keyCfg);

    cout << "\n=== Phrase-as-key benchmark (num_vocab_per_key=" << keyCfg.numVocabPerKey
         << ", num_keys_per_doc=" << keyCfg.numKeysPerDoc << ") ===\n";

    KeyEngine keyEngine;
    auto      keyBuildStart = Clock::now();
    keyEngine.build(keyCorpus.keysPerDoc);
    double keyBuildMs = chrono::duration<double, milli>(Clock::now() - keyBuildStart).count();

    PisaEngine keyPisaEngine;
    auto       keyPisaBuildStart = Clock::now();
    keyPisaEngine.build(keyCorpus.docs);
    double keyPisaBuildMs = chrono::duration<double, milli>(Clock::now() - keyPisaBuildStart).count();

    XapianEngine keyXapianEngine;
    auto         keyXapianBuildStart = Clock::now();
    keyXapianEngine.build(keyCorpus.docs);
    double keyXapianBuildMs = chrono::duration<double, milli>(Clock::now() - keyXapianBuildStart).count();

    cout << "Index build time: key_engine=" << fixed << setprecision(2) << keyBuildMs << " ms, pisa=" << keyPisaBuildMs
         << " ms, xapian=" << keyXapianBuildMs << " ms\n\n";

    vector<string> keyQueries = buildKeyQueries(keyCorpus.keysPerDoc);

    cout << left << setw(26) << "Key" << right << setw(10) << "Key p50" << setw(10) << "Key p95" << setw(10)
         << "Key QPS" << setw(10) << "PISA p50" << setw(10) << "PISA p95" << setw(10) << "PISA QPS" << setw(11)
         << "Xapian p50" << setw(11) << "Xapian p95" << setw(11) << "Xapian QPS" << setw(7) << "Key#" << setw(8)
         << "PISA#" << setw(9) << "Xapian#"
         << "\n";
    cout << left << setw(26) << "" << right << setw(10) << "(ms)" << setw(10) << "(ms)" << setw(10) << "" << setw(10)
         << "(ms)" << setw(10) << "(ms)" << setw(10) << "" << setw(11) << "(ms)" << setw(11) << "(ms)" << setw(11)
         << "" << setw(7) << "" << setw(8) << "" << setw(9) << "" << "\n";
    cout << string(149, '-') << "\n";

    for (const string& key : keyQueries)
    {
        string                 andQuery = keyToAndQuery(key);
        unique_ptr<QueryNode> ast      = QueryParser::parse(andQuery);

        auto [keyResult, keyStats] = timedEvaluate([&] { return keyEngine.evaluate(key); }, args.reps);
        auto [pisaResult, pisaStats] = timedEvaluate([&] { return keyPisaEngine.evaluate(*ast); }, args.reps);
        auto [xapianResult, xapianStats] =
            timedEvaluate([&] { return keyXapianEngine.evaluate(*ast); }, args.reps);

        double keyQps    = 1e6 / keyStats.p50Us;
        double pisaQps    = 1e6 / pisaStats.p50Us;
        double xapianQps  = 1e6 / xapianStats.p50Us;

        string displayKey = key.size() > 24 ? key.substr(0, 21) + "..." : key;
        cout << left << setw(26) << displayKey << right << setw(10) << fixed << setprecision(4)
             << keyStats.p50Us / 1000.0 << setw(10) << keyStats.p95Us / 1000.0 << setw(10) << setprecision(0)
             << keyQps << setw(10) << setprecision(4) << pisaStats.p50Us / 1000.0 << setw(10)
             << pisaStats.p95Us / 1000.0 << setw(10) << setprecision(0) << pisaQps << setw(11) << setprecision(4)
             << xapianStats.p50Us / 1000.0 << setw(11) << xapianStats.p95Us / 1000.0 << setw(11)
             << setprecision(0) << xapianQps << setw(7) << keyResult.size() << setw(8) << pisaResult.size()
             << setw(9) << xapianResult.size() << "\n";
    }

    cout << "\nNote: Key# is expected to be <= PISA#/Xapian# -- the key lookup only matches docs with\n"
         << "the exact key, while the AND query matches any doc containing all the key's terms\n"
         << "anywhere (possibly split across different keys).\n";

    return 0;
}
