// PISA-style engine for the benchmark: the same TermDictionary + posting-list
// design as ../pisa/main.cpp, but evaluating the shared QueryNode AST
// (query_ast.h) instead of a bespoke BooleanQueryEngine, so the exact same
// parsed query tree can be run through both engines in main.cpp.
#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "query_ast.h"

// Term dictionary: maps each unique term string to a dense integer term_id.
// String hashing only ever happens here -- once per term at build time, and
// once per query term at lookup time.
class TermDictionary
{
public:
    int getOrInsert(const std::string& term)
    {
        auto [it, inserted] = termToId_.try_emplace(term, static_cast<int>(termToId_.size()));
        return it->second;
    }

    int lookup(const std::string& term) const
    {
        auto it = termToId_.find(term);
        return it == termToId_.end() ? -1 : it->second;
    }

private:
    std::unordered_map<std::string, int> termToId_;
};

// Inverted index: term_id -> sorted, de-duplicated list of doc ids, stored
// as a flat vector<vector<int>> indexed by term_id (no hashing in the query
// hot path -- only plain integer array accesses + merges).
class PisaEngine
{
public:
    void build(const std::vector<std::string>& docs)
    {
        numDocs_ = static_cast<int>(docs.size());
        for (int docId = 0; docId < numDocs_; docId++)
        {
            for (const std::string& term : tokenize(docs[docId]))
            {
                int termId = dictionary_.getOrInsert(term);
                if (termId >= static_cast<int>(postings_.size()))
                    postings_.resize(termId + 1);
                std::vector<int>& postings = postings_[termId];
                if (postings.empty() || postings.back() != docId)
                {
                    postings.push_back(docId);
                }
            }
        }
    }

    // Evaluates a parsed boolean query AST against this index.
    std::vector<int> evaluate(const QueryNode& node) const
    {
        switch (node.type)
        {
        case NodeType::Term:
        {
            int termId = dictionary_.lookup(node.term);
            return termId < 0 ? std::vector<int> {} : postings_[termId];
        }
        case NodeType::And:
            return intersect(evaluate(*node.left), evaluate(*node.right));
        case NodeType::Or:
            return unionLists(evaluate(*node.left), evaluate(*node.right));
        case NodeType::Not:
            return difference(allDocIds(), evaluate(*node.left));
        }
        return {};
    }

    int numDocs() const { return numDocs_; }

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

    std::vector<int> allDocIds() const
    {
        std::vector<int> all(numDocs_);
        for (int i = 0; i < numDocs_; i++)
            all[i] = i;
        return all;
    }

    static std::vector<int> intersect(const std::vector<int>& a, const std::vector<int>& b)
    {
        std::vector<int> out;
        size_t           i = 0, j = 0;
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

    static std::vector<int> unionLists(const std::vector<int>& a, const std::vector<int>& b)
    {
        std::vector<int> out;
        size_t           i = 0, j = 0;
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

    static std::vector<int> difference(const std::vector<int>& a, const std::vector<int>& b)
    {
        std::vector<int> out;
        size_t           i = 0, j = 0;
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

    TermDictionary                dictionary_;
    std::vector<std::vector<int>> postings_; // indexed by term_id
    int                           numDocs_ = 0;
};
