// Demo: the same toy document set and boolean queries as ../pisa/main.cpp,
// this time using the real Xapian library (https://xapian.org) instead of a
// hand-rolled inverted index. Xapian natively supports nested boolean query
// trees (AND / OR / AND_NOT / ...) and can be configured to skip relevance
// ranking entirely via Xapian::BoolWeight, so no extra glue code is needed
// for pure boolean retrieval -- unlike PISA, where we had to write our own
// BooleanQueryEngine on top of flat AND/OR primitives.
//
// Build: see Makefile (requires `xapian-config` on PATH, i.e. libxapian-dev
// or a from-source install under /usr/local).

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <xapian.h>

using namespace std;

// -------------
// Configuration (same corpus as the PISA demo, for a direct comparison)
const vector<string> kDocuments = {
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day",
    "a quick fox and a quick dog race in the park",
    "machine learning models rank documents by relevance",
    "an inverted index maps terms to the documents that contain them",
    "pisa is a c++ library for efficient inverted indexes and query processing",
};

static vector<string> tokenize(const string& text)
{
    vector<string> tokens;
    stringstream   ss(text);
    string         word;
    while (ss >> word)
        tokens.push_back(word);
    return tokens;
}

// Builds an in-memory Xapian database and indexes kDocuments. Xapian handles
// the term dictionary and posting lists internally -- we only hand it terms.
static Xapian::WritableDatabase buildIndex()
{
    Xapian::WritableDatabase db(string(), Xapian::DB_BACKEND_INMEMORY);
    for (size_t docId = 0; docId < kDocuments.size(); docId++)
    {
        Xapian::Document doc;
        doc.set_data(kDocuments[docId]);
        for (const string& term : tokenize(kDocuments[docId]))
        {
            doc.add_term(term); // Xapian dedupes/accumulates positions per term internally
        }
        db.add_document(doc); // Xapian assigns sequential docids starting at 1
    }
    return db;
}

// Runs `query` against `db` with pure boolean matching (no relevance
// ranking: Xapian::BoolWeight makes every match weigh 0, so results come
// back in whatever order the matcher produces, not by score).
static void runBooleanQuery(const Xapian::Database& db, const Xapian::Query& query, const string& label)
{
    Xapian::Enquire enquire(db);
    enquire.set_query(query);
    enquire.set_weighting_scheme(Xapian::BoolWeight()); // boolean-only, no ranking

    Xapian::MSet matches = enquire.get_mset(0, db.get_doccount());

    cout << label << ": ";
    if (matches.empty())
    {
        cout << "(no matches)\n";
        return;
    }
    for (auto it = matches.begin(); it != matches.end(); ++it)
        cout << (*it - 1) << " "; // back to 0-based doc id
    cout << "\n";
    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        cout << "  [" << (*it - 1) << "] " << it.get_document().get_data() << "\n";
    }
}

int main()
{
    Xapian::WritableDatabase db = buildIndex();
    cout << "Indexed " << db.get_doccount() << " documents.\n\n";

    // Flat AND / OR, built from a vector of single-term queries, mirroring
    // the PISA demo's queryAnd/queryOr.
    vector<Xapian::Query> andTerms = { Xapian::Query("quick"), Xapian::Query("dog") };
    Xapian::Query         queryAnd(Xapian::Query::OP_AND, andTerms.begin(), andTerms.end());
    runBooleanQuery(db, queryAnd, "AND query [quick, dog]");
    cout << "\n";

    vector<Xapian::Query> orTerms = { Xapian::Query("index"), Xapian::Query("pisa") };
    Xapian::Query         queryOr(Xapian::Query::OP_OR, orTerms.begin(), orTerms.end());
    runBooleanQuery(db, queryOr, "OR query [index, pisa]");
    cout << "\n";

    vector<Xapian::Query> and3Terms = { Xapian::Query("quick"), Xapian::Query("dog"), Xapian::Query("park") };
    Xapian::Query         queryAnd3(Xapian::Query::OP_AND, and3Terms.begin(), and3Terms.end());
    runBooleanQuery(db, queryAnd3, "AND query [quick, dog, park]");
    cout << "\n";

    cout << "--- complex boolean queries (built directly as Xapian::Query trees) ---\n\n";

    // (quick AND dog) OR (NOT fox)
    // Xapian has no standalone NOT; "not fox" is expressed as
    // OP_AND_NOT(match-everything, fox), i.e. all docs minus docs with "fox".
    vector<Xapian::Query> quickAndDogTerms = { Xapian::Query("quick"), Xapian::Query("dog") };
    Xapian::Query         quickAndDog(Xapian::Query::OP_AND, quickAndDogTerms.begin(), quickAndDogTerms.end());
    Xapian::Query         notFox(Xapian::Query::OP_AND_NOT, Xapian::Query::MatchAll, Xapian::Query("fox"));
    runBooleanQuery(db, Xapian::Query(Xapian::Query::OP_OR, quickAndDog, notFox), "(quick AND dog) OR (NOT fox)");
    cout << "\n";

    // quick AND NOT park
    Xapian::Query quickAndNotPark(Xapian::Query::OP_AND_NOT, Xapian::Query("quick"), Xapian::Query("park"));
    runBooleanQuery(db, quickAndNotPark, "quick AND NOT park");
    cout << "\n";

    // (index OR pisa) AND NOT library
    vector<Xapian::Query> indexOrPisaTerms = { Xapian::Query("index"), Xapian::Query("pisa") };
    Xapian::Query         indexOrPisa(Xapian::Query::OP_OR, indexOrPisaTerms.begin(), indexOrPisaTerms.end());
    Xapian::Query         result3(Xapian::Query::OP_AND_NOT, indexOrPisa, Xapian::Query("library"));
    runBooleanQuery(db, result3, "(index OR pisa) AND NOT library");
    cout << "\n";

    // quick AND (dog OR fox) AND NOT lazy
    vector<Xapian::Query> dogOrFoxTerms = { Xapian::Query("dog"), Xapian::Query("fox") };
    Xapian::Query         dogOrFox(Xapian::Query::OP_OR, dogOrFoxTerms.begin(), dogOrFoxTerms.end());
    vector<Xapian::Query> quickAndDogOrFoxTerms = { Xapian::Query("quick"), dogOrFox };
    Xapian::Query quickAndDogOrFox(Xapian::Query::OP_AND, quickAndDogOrFoxTerms.begin(), quickAndDogOrFoxTerms.end());
    Xapian::Query result4(Xapian::Query::OP_AND_NOT, quickAndDogOrFox, Xapian::Query("lazy"));
    runBooleanQuery(db, result4, "quick AND (dog OR fox) AND NOT lazy");

    return 0;
}
