// Benchmark: generates a synthetic corpus (seed phrases sampled into docs,
// so phrases naturally recur across docs), builds PISA-style, Xapian, and
// KeyEngine indexes over it, runs an identical set of boolean queries
// (from simple single-term to multi-level nested AND/OR/NOT) against PISA
// and Xapian, and reports result count + latency for each engine/query
// pair. Also cross-checks that both engines agree on the result doc-id
// set, as a correctness sanity check.
//
// A second section benchmarks a "phrase-as-key" scenario: a simple
// unordered_map<string, vector<docId>> KeyEngine where a whole multi-term
// phrase (e.g. "vocA vocB vocC") is the lookup unit, compared against PISA
// and Xapian answering the equivalent classic AND query over the same
// corpus.
//
// Usage:
//   ./benchmark [--num_docs=N] [--num_char_per_vocab=N] [--phrase_card=N]
//               [--reps=N] [--num_vocab_per_phrase=N] [--num_phrases_per_doc=N]
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

#ifdef HAVE_CUDA
#include "cuda_copy.h"
#endif

using namespace std;
using Clock = chrono::high_resolution_clock;

struct Args
{
    CorpusConfig corpus;
    int          reps    = 20;    // repetitions per query; reports p50/p95 latency across these reps
    bool         gpuCopy = false; // if set, also measure a real cudaMalloc+cudaMemcpy D2H copy per query
};

static void printUsage(const char* program)
{
    Args defaults; // show default values alongside each flag
    cout << "Usage: " << program << " [OPTIONS]\n\n"
         << "Generates a synthetic corpus (built from a fixed pool of seed phrases,\n"
         << "reused across docs) and benchmarks PISA, Xapian, and a phrase-as-key\n"
         << "hash-map engine (KeyEngine) on it.\n\n"
         << "Options:\n"
         << "  -d, --num_docs=N              Number of documents to generate (default: " << defaults.corpus.numDocs
         << ")\n"
         << "  -l, --num_char_per_vocab=N    Character length of each distinct vocabulary term (default: "
         << defaults.corpus.numCharPerVocab << ")\n"
         << "  -k, --phrase_card=N           Phrase cardinality, i.e. # of distinct seed phrases (default: "
         << defaults.corpus.phraseCard << ")\n"
         << "  -r, --reps=N                  Repetitions per query; reports p50/p95 latency across them (default: "
         << defaults.reps << ")\n"
         << "  -p, --num_vocab_per_phrase=N  Terms per seed phrase (default: " << defaults.corpus.numVocabPerPhrase
         << ")\n"
         << "  -n, --num_phrases_per_doc=N   Seed phrases sampled into each doc (default: "
         << defaults.corpus.numPhrasesPerDoc << ")\n"
         << "  -g, --gpu_copy                Also measure a real cudaMalloc+cudaMemcpy D2H copy per\n"
         << "                                 query (requires building with nvcc/CUDA and running on a\n"
         << "                                 CUDA-capable GPU; see Makefile)\n"
         << "  -h, --help                    Show this help message and exit\n\n"
         << "Example:\n"
         << "  " << program
         << " --num_docs=200000 --phrase_card=5000 --num_char_per_vocab=8 --reps=7 --num_vocab_per_phrase=3"
         << " --num_phrases_per_doc=5\n"
         << "  " << program << " -d 200000 -k 5000 -l 8 -r 7 -p 3 -n 5\n";
}

static Args parseArgs(int argc, char** argv)
{
    Args args;
    // Maps a flag (long or short form) to the Args field it sets.
    auto apply = [&](const string& key, int value)
    {
        if (key == "num_docs" || key == "d")
            args.corpus.numDocs = value;
        else if (key == "num_char_per_vocab" || key == "l")
            args.corpus.numCharPerVocab = value;
        else if (key == "phrase_card" || key == "k")
            args.corpus.phraseCard = value;
        else if (key == "reps" || key == "r")
            args.reps = value;
        else if (key == "num_vocab_per_phrase" || key == "p")
            args.corpus.numVocabPerPhrase = value;
        else if (key == "num_phrases_per_doc" || key == "n")
            args.corpus.numPhrasesPerDoc = value;
    };

    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            exit(0);
        }
        if (arg == "-g" || arg == "--gpu_copy")
        {
            args.gpuCopy = true;
            continue;
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

// Picks a handful of distinct phrases (in order of first appearance) from
// the corpus to use as phrase-as-key benchmark queries -- guaranteed to
// exist, so results are non-empty.
static vector<string> buildKeyQueries(const vector<vector<string>>& phrasesPerDoc, int maxQueries = 5)
{
    vector<string>        queries;
    unordered_set<string> seen;
    for (const auto& phrases : phrasesPerDoc)
    {
        for (const string& phrase : phrases)
        {
            if (seen.insert(phrase).second)
            {
                queries.push_back(phrase);
                if (static_cast<int>(queries.size()) >= maxQueries)
                    return queries;
            }
        }
    }
    return queries;
}

// Splits a phrase string ("vocA vocB vocC") into the equivalent classic
// boolean AND query ("vocA AND vocB AND vocC") that PISA/Xapian can evaluate.
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
    size_t n    = sortedValues.size();
    size_t rank = static_cast<size_t>(ceil(p / 100.0 * static_cast<double>(n)));
    rank        = max<size_t>(rank, 1);
    rank        = min(rank, n);
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

// Times copying a result vector<int> out of the engine's ownership into a
// fresh vector, repeated `reps` times. Models the "D2H"-style cost of
// transferring a query's result doc-id list to the caller, isolated from
// the lookup/merge compute time measured by timedEvaluate() above. All
// three engines here are CPU-only (no device memory), so this is a
// same-host copy, but it's still useful to see as its own line item since
// it scales with result size, not query complexity.
static LatencyStats timedCopy(const vector<int>& src, int reps)
{
    // volatile sink: forces the compiler to actually materialize each copy
    // instead of proving it unused and eliding it.
    static volatile size_t sink = 0;

    vector<double> latenciesUs;
    for (int r = 0; r < reps; r++)
    {
        auto        start = Clock::now();
        vector<int> copy(src);
        auto        end = Clock::now();
        sink            = copy.size();
        latenciesUs.push_back(chrono::duration<double, micro>(end - start).count());
    }
    sort(latenciesUs.begin(), latenciesUs.end());
    LatencyStats stats;
    stats.p50Us = percentile(latenciesUs, 50.0);
    stats.p95Us = percentile(latenciesUs, 95.0);
    return stats;
}

// Wraps the real cudaMalloc+cudaMemcpy D2H helper (cuda_copy.cu, only built
// when nvcc/CUDA is available) into a LatencyStats, so callers don't need
// to care about the HAVE_CUDA ifdef at every call site. Only called when
// args.gpuCopy is set, which main() already guards to require HAVE_CUDA.
#ifdef HAVE_CUDA
static LatencyStats timedGpuCopy(const vector<int>& src, int reps)
{
    GpuCopyStats gpu = timedCudaD2HCopy(src, reps);
    LatencyStats stats;
    stats.p50Us = gpu.p50Us;
    stats.p95Us = gpu.p95Us;
    return stats;
}
#endif

int main(int argc, char** argv)
{
    Args                args = parseArgs(argc, argv);
    const CorpusConfig& cfg  = args.corpus;

#ifndef HAVE_CUDA
    if (args.gpuCopy)
    {
        cerr << "--gpu_copy requires building with nvcc/CUDA (see Makefile: nvcc must be on PATH),\n"
             << "but this binary was built without CUDA support. Rebuild on a machine with the CUDA\n"
             << "toolkit installed, or omit --gpu_copy.\n";
        return 1;
    }
#endif

    cout << "Corpus config: num_docs=" << cfg.numDocs << " num_char_per_vocab=" << cfg.numCharPerVocab
         << " phrase_card=" << cfg.phraseCard << " num_vocab_per_phrase=" << cfg.numVocabPerPhrase
         << " num_phrases_per_doc=" << cfg.numPhrasesPerDoc << "\n\n";

    Corpus corpus = generateCorpus(cfg);

    PisaEngine pisaEngine;
    auto       pisaBuildStart = Clock::now();
    pisaEngine.build(corpus.docs);
    double pisaBuildMs = chrono::duration<double, milli>(Clock::now() - pisaBuildStart).count();

    XapianEngine xapianEngine;
    auto         xapianBuildStart = Clock::now();
    xapianEngine.build(corpus.docs);
    double xapianBuildMs = chrono::duration<double, milli>(Clock::now() - xapianBuildStart).count();

    KeyEngine keyEngine;
    auto      keyBuildStart = Clock::now();
    keyEngine.build(corpus.phrasesPerDoc);
    double keyBuildMs = chrono::duration<double, milli>(Clock::now() - keyBuildStart).count();

    cout << "Index build time: pisa=" << fixed << setprecision(2) << pisaBuildMs << " ms, xapian=" << xapianBuildMs
         << " ms, key_engine=" << keyBuildMs << " ms\n";
    cout << "Indexed docs: pisa=" << pisaEngine.numDocs() << ", xapian=" << xapianEngine.numDocs() << "\n\n";

    vector<string> queries = buildQueries(corpus.vocab);

    // Headers are built into strings first (rather than streamed straight to
    // cout) so the separator line below can be sized to match exactly,
    // regardless of whether the GPU columns are present.
    ostringstream header1, header2;
    header1 << left << setw(38) << "Query" << right << setw(10) << "PISA p50" << setw(10) << "PISA p95" << setw(11)
            << "PISA QPS" << setw(11) << "PISA Copy";
    header2 << left << setw(38) << "" << right << setw(10) << "(ms)" << setw(10) << "(ms)" << setw(11) << "" << setw(11)
            << "p50(ms)";
    if (args.gpuCopy)
    {
        header1 << setw(13) << "PISA GPUCopy";
        header2 << setw(13) << "p50(ms)";
    }
    header1 << setw(12) << "Xapian p50" << setw(12) << "Xapian p95" << setw(11) << "Xapian QPS" << setw(13)
            << "Xapian Copy";
    header2 << setw(12) << "(ms)" << setw(12) << "(ms)" << setw(11) << "" << setw(13) << "p50(ms)";
    if (args.gpuCopy)
    {
        header1 << setw(15) << "Xapian GPUCopy";
        header2 << setw(15) << "p50(ms)";
    }
    header1 << setw(9) << "PISA#" << setw(9) << "Xapian#" << setw(7) << "Match";

    cout << header1.str() << "\n" << header2.str() << "\n";
    cout << string(header1.str().size(), '-') << "\n";

    for (const string& query : queries)
    {
        unique_ptr<QueryNode> ast = QueryParser::parse(query);

        auto [pisaResult, pisaStats]     = timedEvaluate([&] { return pisaEngine.evaluate(*ast); }, args.reps);
        auto [xapianResult, xapianStats] = timedEvaluate([&] { return xapianEngine.evaluate(*ast); }, args.reps);

        LatencyStats pisaCopyStats   = timedCopy(pisaResult, args.reps);
        LatencyStats xapianCopyStats = timedCopy(xapianResult, args.reps);
#ifdef HAVE_CUDA
        LatencyStats pisaGpuCopyStats, xapianGpuCopyStats;
        if (args.gpuCopy)
        {
            pisaGpuCopyStats   = timedGpuCopy(pisaResult, args.reps);
            xapianGpuCopyStats = timedGpuCopy(xapianResult, args.reps);
        }
#endif

        // QPS here means single-threaded queries/sec implied by one query's
        // p50 latency (1e6 / p50_latency_us), not a multi-threaded
        // throughput measurement.
        double pisaQps   = 1e6 / pisaStats.p50Us;
        double xapianQps = 1e6 / xapianStats.p50Us;

        bool match = (pisaResult == xapianResult);

        string displayQuery = query.size() > 36 ? query.substr(0, 33) + "..." : query;
        cout << left << setw(38) << displayQuery << right << setw(10) << fixed << setprecision(4)
             << pisaStats.p50Us / 1000.0 << setw(10) << pisaStats.p95Us / 1000.0 << setw(11) << setprecision(0)
             << pisaQps << setw(11) << setprecision(4) << pisaCopyStats.p50Us / 1000.0;
#ifdef HAVE_CUDA
        if (args.gpuCopy)
        {
            cout << setw(13) << setprecision(4) << pisaGpuCopyStats.p50Us / 1000.0;
        }
#endif
        cout << setw(12) << xapianStats.p50Us / 1000.0 << setw(12) << xapianStats.p95Us / 1000.0 << setw(11)
             << setprecision(0) << xapianQps << setw(13) << setprecision(4) << xapianCopyStats.p50Us / 1000.0;
#ifdef HAVE_CUDA
        if (args.gpuCopy)
        {
            cout << setw(15) << setprecision(4) << xapianGpuCopyStats.p50Us / 1000.0;
        }
#endif
        cout << setw(9) << pisaResult.size() << setw(9) << xapianResult.size() << setw(7) << (match ? "yes" : "NO")
             << "\n";

        if (!match)
        {
            cerr << "  !! Mismatch for query: " << query << "\n";
        }
    }

    // --- Phrase-as-key benchmark ---------------------------------------
    // Uses the *same* corpus and PISA/Xapian indexes built above. KeyEngine
    // looks a whole phrase up as one hash-map entry; PISA/Xapian answer the
    // "same" query as a classic boolean AND of the phrase's individual
    // terms. The two are NOT expected to return identical doc sets: AND
    // matches any doc containing all terms anywhere (even split across
    // different phrases), while the phrase lookup only matches docs having
    // that exact phrase.
    cout << "\n=== Phrase-as-key benchmark (num_vocab_per_phrase=" << cfg.numVocabPerPhrase
         << ", num_phrases_per_doc=" << cfg.numPhrasesPerDoc << ") ===\n\n";

    vector<string> keyQueries = buildKeyQueries(corpus.phrasesPerDoc);

    ostringstream keyHeader1, keyHeader2;
    keyHeader1 << left << setw(26) << "Key" << right << setw(10) << "Key p50" << setw(10) << "Key p95" << setw(10)
               << "Key QPS" << setw(10) << "Key Copy";
    keyHeader2 << left << setw(26) << "" << right << setw(10) << "(ms)" << setw(10) << "(ms)" << setw(10) << ""
               << setw(10) << "p50(ms)";
    if (args.gpuCopy)
    {
        keyHeader1 << setw(12) << "Key GPUCopy";
        keyHeader2 << setw(12) << "p50(ms)";
    }
    keyHeader1 << setw(10) << "PISA p50" << setw(10) << "PISA p95" << setw(10) << "PISA QPS" << setw(10) << "PISA Copy";
    keyHeader2 << setw(10) << "(ms)" << setw(10) << "(ms)" << setw(10) << "" << setw(10) << "p50(ms)";
    if (args.gpuCopy)
    {
        keyHeader1 << setw(13) << "PISA GPUCopy";
        keyHeader2 << setw(13) << "p50(ms)";
    }
    keyHeader1 << setw(11) << "Xapian p50" << setw(11) << "Xapian p95" << setw(11) << "Xapian QPS" << setw(11)
               << "XapianCopy";
    keyHeader2 << setw(11) << "(ms)" << setw(11) << "(ms)" << setw(11) << "" << setw(11) << "p50(ms)";
    if (args.gpuCopy)
    {
        keyHeader1 << setw(15) << "Xapian GPUCopy";
        keyHeader2 << setw(15) << "p50(ms)";
    }
    keyHeader1 << setw(7) << "Key#" << setw(8) << "PISA#" << setw(9) << "Xapian#";

    cout << keyHeader1.str() << "\n" << keyHeader2.str() << "\n";
    cout << string(keyHeader1.str().size(), '-') << "\n";

    for (const string& key : keyQueries)
    {
        string                andQuery = keyToAndQuery(key);
        unique_ptr<QueryNode> ast      = QueryParser::parse(andQuery);

        auto [keyResult, keyStats]       = timedEvaluate([&] { return keyEngine.evaluate(key); }, args.reps);
        auto [pisaResult, pisaStats]     = timedEvaluate([&] { return pisaEngine.evaluate(*ast); }, args.reps);
        auto [xapianResult, xapianStats] = timedEvaluate([&] { return xapianEngine.evaluate(*ast); }, args.reps);

        LatencyStats keyCopyStats    = timedCopy(keyResult, args.reps);
        LatencyStats pisaCopyStats   = timedCopy(pisaResult, args.reps);
        LatencyStats xapianCopyStats = timedCopy(xapianResult, args.reps);
#ifdef HAVE_CUDA
        LatencyStats keyGpuCopyStats, pisaGpuCopyStats, xapianGpuCopyStats;
        if (args.gpuCopy)
        {
            keyGpuCopyStats    = timedGpuCopy(keyResult, args.reps);
            pisaGpuCopyStats   = timedGpuCopy(pisaResult, args.reps);
            xapianGpuCopyStats = timedGpuCopy(xapianResult, args.reps);
        }
#endif

        double keyQps    = 1e6 / keyStats.p50Us;
        double pisaQps   = 1e6 / pisaStats.p50Us;
        double xapianQps = 1e6 / xapianStats.p50Us;

        string displayKey = key.size() > 24 ? key.substr(0, 21) + "..." : key;
        cout << left << setw(26) << displayKey << right << setw(10) << fixed << setprecision(4)
             << keyStats.p50Us / 1000.0 << setw(10) << keyStats.p95Us / 1000.0 << setw(10) << setprecision(0) << keyQps
             << setw(10) << setprecision(4) << keyCopyStats.p50Us / 1000.0;
#ifdef HAVE_CUDA
        if (args.gpuCopy)
        {
            cout << setw(12) << setprecision(4) << keyGpuCopyStats.p50Us / 1000.0;
        }
#endif
        cout << setw(10) << pisaStats.p50Us / 1000.0 << setw(10) << pisaStats.p95Us / 1000.0 << setw(10)
             << setprecision(0) << pisaQps << setw(10) << setprecision(4) << pisaCopyStats.p50Us / 1000.0;
#ifdef HAVE_CUDA
        if (args.gpuCopy)
        {
            cout << setw(13) << setprecision(4) << pisaGpuCopyStats.p50Us / 1000.0;
        }
#endif
        cout << setw(11) << xapianStats.p50Us / 1000.0 << setw(11) << xapianStats.p95Us / 1000.0 << setw(11)
             << setprecision(0) << xapianQps << setw(11) << setprecision(4) << xapianCopyStats.p50Us / 1000.0;
#ifdef HAVE_CUDA
        if (args.gpuCopy)
        {
            cout << setw(15) << setprecision(4) << xapianGpuCopyStats.p50Us / 1000.0;
        }
#endif
        cout << setw(7) << keyResult.size() << setw(8) << pisaResult.size() << setw(9) << xapianResult.size() << "\n";
    }

    cout << "\nNote: Key# is expected to be <= PISA#/Xapian# -- the key lookup only matches docs with\n"
         << "the exact key, while the AND query matches any doc containing all the key's terms\n"
         << "anywhere (possibly split across different keys).\n";

    return 0;
}
