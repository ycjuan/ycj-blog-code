// Demo: building and querying an inverted index, modeled after the core
// concepts used by the real PISA library (https://github.com/pisa-engine/pisa):
//   - a term -> term_id dictionary (string hashing happens exactly once per
//     term, at dictionary build/lookup time)
//   - per-term posting lists of sorted doc ids, stored as a plain
//     vector<vector<int>> indexed by *integer* term_id (delta-friendly
//     layout, no string hashing at all during query processing)
//   - conjunctive (AND) and disjunctive (OR) query processing via a simple
//     merge over posting lists
//
// This is a small, self-contained, educational re-implementation (no
// external dependencies) so it builds with a single g++ invocation. PISA
// itself adds on top of these ideas: SIMD-friendly compression (streamvbyte,
// MaskedVByte, simdcomp, QMX...), block-max indexes, WAND/BlockMax-WAND
// query processing, and on-disk binary formats for large collections.

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// -------------
// Configuration
const vector<string> kDocuments = {
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day",
    "a quick fox and a quick dog race in the park",
    "machine learning models rank documents by relevance",
    "an inverted index maps terms to the documents that contain them",
    "pisa is a c++ library for efficient inverted indexes and query processing",
};

// -------------
// Term dictionary: maps each unique term string to a dense integer term_id.
// String hashing only ever happens here -- once per term at build time, and
// once per query term at lookup time. Everything downstream (posting-list
// storage, intersection, union) operates purely on ints.
class TermDictionary
{
public:
    // Returns the term_id for `term`, assigning a new one if unseen.
    int getOrInsert(const string& term)
    {
        auto [it, inserted] = termToId_.try_emplace(term, static_cast<int>(terms_.size()));
        if (inserted)
            terms_.push_back(term);
        return it->second;
    }

    // Returns the term_id for `term`, or -1 if the term was never indexed.
    int lookup(const string& term) const
    {
        auto it = termToId_.find(term);
        return it == termToId_.end() ? -1 : it->second;
    }

    int size() const { return static_cast<int>(terms_.size()); }

private:
    unordered_map<string, int> termToId_; // string -> term_id (hashing happens here only)
    vector<string>             terms_;    // term_id -> string (for debugging/printing)
};

// -------------
// Inverted index: term_id -> sorted, de-duplicated list of doc ids.
// Posting lists are stored as a flat vector<vector<int>> indexed by
// term_id, so lookups during query processing are O(1) array accesses
// instead of string hashes.
class InvertedIndex
{
public:
    void build(const vector<string>& docs)
    {
        numDocs_ = static_cast<int>(docs.size());
        for (int docId = 0; docId < numDocs_; docId++)
        {
            for (const string& term : tokenize(docs[docId]))
            {
                int termId = dictionary_.getOrInsert(term);
                if (termId >= static_cast<int>(postings_.size()))
                    postings_.resize(termId + 1);
                vector<int>& postings = postings_[termId];
                if (postings.empty() || postings.back() != docId)
                {
                    postings.push_back(docId);
                }
            }
        }
    }

    // Conjunctive (AND) query: docs containing every term.
    vector<int> queryAnd(const vector<string>& terms) const
    {
        vector<vector<int>> lists = postingLists(terms);
        if (lists.empty())
            return {};

        // Start from the shortest list to minimize work, then intersect.
        sort(lists.begin(),
             lists.end(),
             [](const vector<int>& a, const vector<int>& b) { return a.size() < b.size(); });

        vector<int> result = lists[0];
        for (size_t i = 1; i < lists.size() && !result.empty(); i++)
        {
            result = intersect(result, lists[i]);
        }
        return result;
    }

    // Disjunctive (OR) query: docs containing at least one term.
    vector<int> queryOr(const vector<string>& terms) const
    {
        vector<int> result;
        for (const vector<int>& postings : postingLists(terms))
        {
            result = unionLists(result, postings);
        }
        return result;
    }

    // Posting list for a single term (empty if the term was never indexed).
    vector<int> postingsFor(const string& term) const
    {
        int termId = dictionary_.lookup(term);
        return termId < 0 ? vector<int> {} : postings_[termId];
    }

    // All doc ids [0, numDocs), used to evaluate NOT as a complement.
    vector<int> allDocIds() const
    {
        vector<int> all(numDocs_);
        for (int i = 0; i < numDocs_; i++)
            all[i] = i;
        return all;
    }

    int numDocs() const { return numDocs_; }

    static vector<int> intersect(const vector<int>& a, const vector<int>& b)
    {
        vector<int> out;
        size_t      i = 0, j = 0;
        while (i < a.size() && j < b.size())
        {
            if (a[i] == b[j])
            {
                out.push_back(a[i]);
                i++;
                j++;
            }
            else if (a[i] < b[j])
                i++;
            else
                j++;
        }
        return out;
    }

    static vector<int> unionLists(const vector<int>& a, const vector<int>& b)
    {
        vector<int> out;
        size_t      i = 0, j = 0;
        while (i < a.size() && j < b.size())
        {
            if (a[i] == b[j])
            {
                out.push_back(a[i]);
                i++;
                j++;
            }
            else if (a[i] < b[j])
                out.push_back(a[i++]);
            else
                out.push_back(b[j++]);
        }
        while (i < a.size())
            out.push_back(a[i++]);
        while (j < b.size())
            out.push_back(b[j++]);
        return out;
    }

    // a AND NOT b, i.e. elements of `a` not present in `b`.
    static vector<int> difference(const vector<int>& a, const vector<int>& b)
    {
        vector<int> out;
        size_t      i = 0, j = 0;
        while (i < a.size() && j < b.size())
        {
            if (a[i] == b[j])
            {
                i++;
                j++;
            }
            else if (a[i] < b[j])
                out.push_back(a[i++]);
            else
                j++;
        }
        while (i < a.size())
            out.push_back(a[i++]);
        return out;
    }

private:
    static vector<string> tokenize(const string& text)
    {
        vector<string> tokens;
        stringstream   ss(text);
        string         word;
        while (ss >> word)
            tokens.push_back(word);
        return tokens;
    }

    // Translates query terms to term_ids (one hash lookup per term), then
    // fetches posting lists by plain integer index -- no further hashing.
    vector<vector<int>> postingLists(const vector<string>& terms) const
    {
        vector<vector<int>> lists;
        for (const string& term : terms)
        {
            int termId = dictionary_.lookup(term);
            if (termId < 0)
                return {}; // term unseen -> no matches for AND; caller handles OR separately
            lists.push_back(postings_[termId]);
        }
        return lists;
    }

    TermDictionary      dictionary_;
    vector<vector<int>> postings_; // indexed by term_id
    int                 numDocs_ = 0;
};

// -------------
// Boolean query engine: parses and evaluates arbitrary nested expressions
// with AND / OR / NOT and parentheses, e.g.:
//   "(quick AND dog) OR (NOT fox)"
//   "quick AND NOT park"
// Grammar (standard precedence: NOT > AND > OR):
//   expr   := andExpr (OR andExpr)*
//   andExpr:= notExpr (AND notExpr)*
//   notExpr:= NOT notExpr | primary
//   primary:= TERM | '(' expr ')'
class BooleanQueryEngine
{
public:
    explicit BooleanQueryEngine(const InvertedIndex& index)
        : index_(index)
    {
    }

    vector<int> evaluate(const string& query)
    {
        tokens_            = tokenize(query);
        pos_               = 0;
        vector<int> result = parseExpr();
        return result;
    }

private:
    vector<int> parseExpr() // OR
    {
        vector<int> result = parseAnd();
        while (peek() == "OR")
        {
            advance();
            result = InvertedIndex::unionLists(result, parseAnd());
        }
        return result;
    }

    vector<int> parseAnd()
    {
        vector<int> result = parseNot();
        while (peek() == "AND")
        {
            advance();
            result = InvertedIndex::intersect(result, parseNot());
        }
        return result;
    }

    vector<int> parseNot()
    {
        if (peek() == "NOT")
        {
            advance();
            return InvertedIndex::difference(index_.allDocIds(), parseNot());
        }
        return parsePrimary();
    }

    vector<int> parsePrimary()
    {
        if (peek() == "(")
        {
            advance();
            vector<int> result = parseExpr();
            if (peek() == ")")
                advance();
            return result;
        }
        string term = peek();
        advance();
        return index_.postingsFor(term);
    }

    const string& peek() const
    {
        static const string kEnd = "";
        return pos_ < tokens_.size() ? tokens_[pos_] : kEnd;
    }

    void advance() { pos_++; }

    // Splits on whitespace and parenthesis boundaries, lower-cases terms but
    // preserves AND/OR/NOT as-is (query boolean keywords are case-sensitive
    // uppercase, like most boolean search syntaxes).
    static vector<string> tokenize(const string& query)
    {
        vector<string> tokens;
        string         current;
        auto           flush = [&]()
        {
            if (!current.empty())
            {
                tokens.push_back(current);
                current.clear();
            }
        };
        for (char c : query)
        {
            if (c == '(' || c == ')')
            {
                flush();
                tokens.push_back(string(1, c));
            }
            else if (isspace(static_cast<unsigned char>(c)))
            {
                flush();
            }
            else
            {
                current.push_back(c);
            }
        }
        flush();
        return tokens;
    }

    const InvertedIndex& index_;
    vector<string>       tokens_;
    size_t               pos_ = 0;
};

static void printResults(const string& label, const vector<int>& docIds)
{
    cout << label << ": ";
    if (docIds.empty())
    {
        cout << "(no matches)\n";
        return;
    }
    for (int docId : docIds)
        cout << docId << " ";
    cout << "\n";
    for (int docId : docIds)
    {
        cout << "  [" << docId << "] " << kDocuments[docId] << "\n";
    }
}

int main()
{
    InvertedIndex index;
    index.build(kDocuments);

    cout << "Indexed " << index.numDocs() << " documents.\n\n";

    printResults("AND query [quick, dog]", index.queryAnd({ "quick", "dog" }));
    cout << "\n";
    printResults("OR query [index, pisa]", index.queryOr({ "index", "pisa" }));
    cout << "\n";
    printResults("AND query [quick, dog, park]", index.queryAnd({ "quick", "dog", "park" }));
    cout << "\n";

    cout << "--- complex boolean queries ---\n\n";
    BooleanQueryEngine engine(index);

    vector<string> booleanQueries = {
        "(quick AND dog) OR (NOT fox)",
        "quick AND NOT park",
        "(index OR pisa) AND NOT library",
        "quick AND (dog OR fox) AND NOT lazy",
    };
    for (const string& query : booleanQueries)
    {
        printResults(query, engine.evaluate(query));
        cout << "\n";
    }

    return 0;
}
