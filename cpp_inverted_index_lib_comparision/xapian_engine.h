// Xapian engine for the benchmark: builds an in-memory Xapian database and
// evaluates the shared QueryNode AST (query_ast.h) by translating it
// directly into an Xapian::Query tree (OP_AND / OP_OR / OP_AND_NOT), then
// running it with Xapian::BoolWeight (pure boolean matching, no ranking).
#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <xapian.h>

#include "query_ast.h"

class XapianEngine
{
public:
    XapianEngine()
        : db_(std::string(), Xapian::DB_BACKEND_INMEMORY)
    {
    }

    void build(const std::vector<std::string>& docs)
    {
        for (const std::string& docText : docs)
        {
            Xapian::Document doc;
            for (const std::string& term : tokenize(docText))
                doc.add_term(term);
            db_.add_document(doc);
        }
    }

    // Evaluates a parsed boolean query AST against this index, returning
    // 0-based doc ids sorted ascending (matching PisaEngine's convention).
    std::vector<int> evaluate(const QueryNode& node) const
    {
        Xapian::Query query = toXapianQuery(node);

        Xapian::Enquire enquire(db_);
        enquire.set_query(query);
        enquire.set_weighting_scheme(Xapian::BoolWeight()); // boolean-only, no ranking

        Xapian::MSet matches = enquire.get_mset(0, db_.get_doccount());

        std::vector<int> docIds;
        docIds.reserve(matches.size());
        for (auto it = matches.begin(); it != matches.end(); ++it)
        {
            docIds.push_back(static_cast<int>(*it) - 1); // Xapian ids are 1-based
        }
        std::sort(docIds.begin(), docIds.end());
        return docIds;
    }

    Xapian::doccount numDocs() const { return db_.get_doccount(); }

private:
    static std::vector<std::string> tokenize(const std::string& text)
    {
        std::vector<std::string> tokens;
        std::stringstream        ss(text);
        std::string              word;
        while (ss >> word)
            tokens.push_back(word);
        return tokens;
    }

    static Xapian::Query toXapianQuery(const QueryNode& node)
    {
        switch (node.type)
        {
        case NodeType::Term:
            return Xapian::Query(node.term);
        case NodeType::And:
            return Xapian::Query(Xapian::Query::OP_AND, toXapianQuery(*node.left), toXapianQuery(*node.right));
        case NodeType::Or:
            return Xapian::Query(Xapian::Query::OP_OR, toXapianQuery(*node.left), toXapianQuery(*node.right));
        case NodeType::Not:
            return Xapian::Query(Xapian::Query::OP_AND_NOT, Xapian::Query::MatchAll, toXapianQuery(*node.left));
        }
        return Xapian::Query();
    }

    Xapian::WritableDatabase db_;
};
