// Benchmark: generates a synthetic corpus, builds both the PISA-style and
// Xapian inverted-index engines, runs an identical set of boolean queries
// (from simple single-term to multi-level nested AND/OR/NOT) against both,
// and reports result count + latency for each engine/query pair. Also
// cross-checks that both engines agree on the result doc-id set, as a
// correctness sanity check.
//
// Usage:
//   ./benchmark [--num_docs=N] [--num_vocab_per_doc=N] [--num_char_per_vocab=N]
//               [--vocab_card=N] [--reps=N]
//
// Build: see Makefile (requires `xapian-config` on PATH).

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "corpus_generator.h"
#include "pisa_engine.h"
#include "query_ast.h"
#include "xapian_engine.h"

using namespace std;
using Clock = chrono::high_resolution_clock;

struct Args
{
    CorpusConfig corpus;
    int          reps = 5; // repetitions per query, report the median latency
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
         << "  -r, --reps=N                Repetitions per query; reported latency is the median (default: "
         << defaults.reps << ")\n"
         << "  -h, --help                  Show this help message and exit\n\n"
         << "Example:\n"
         << "  " << program << " --num_docs=200000 --num_vocab_per_doc=30 --vocab_card=5000 --num_char_per_vocab=8"
         << " --reps=7\n"
         << "  " << program << " -d 200000 -v 30 -k 5000 -l 8 -r 7\n";
}

static Args parseArgs(int argc, char** argv)
{
    Args args;
    // Maps a flag (long or short form) to the Args field it sets.
    auto apply = [&](const string& key, int value) {
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

template <typename Engine>
static pair<vector<int>, double> timedEvaluate(const Engine& engine, const QueryNode& ast, int reps)
{
    vector<double> latenciesUs;
    vector<int>    result;
    for (int r = 0; r < reps; r++)
    {
        auto start = Clock::now();
        result     = engine.evaluate(ast);
        auto end   = Clock::now();
        latenciesUs.push_back(chrono::duration<double, micro>(end - start).count());
    }
    sort(latenciesUs.begin(), latenciesUs.end());
    double median = latenciesUs[latenciesUs.size() / 2];
    return { result, median };
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

    cout << left << setw(60) << "Query" << right << setw(12) << "PISA(us)" << setw(12) << "Xapian(us)" << setw(14)
         << "PISA count" << setw(14) << "Xapian count" << setw(10) << "Match"
         << "\n";
    cout << string(122, '-') << "\n";

    for (const string& query : queries)
    {
        unique_ptr<QueryNode> ast = QueryParser::parse(query);

        auto [pisaResult, pisaLatencyUs]     = timedEvaluate(pisaEngine, *ast, args.reps);
        auto [xapianResult, xapianLatencyUs] = timedEvaluate(xapianEngine, *ast, args.reps);

        bool match = (pisaResult == xapianResult);

        string displayQuery = query.size() > 58 ? query.substr(0, 55) + "..." : query;
        cout << left << setw(60) << displayQuery << right << setw(12) << fixed << setprecision(1) << pisaLatencyUs
             << setw(12) << xapianLatencyUs << setw(14) << pisaResult.size() << setw(14) << xapianResult.size()
             << setw(10) << (match ? "yes" : "NO") << "\n";

        if (!match)
        {
            cerr << "  !! Mismatch for query: " << query << "\n";
        }
    }

    return 0;
}
