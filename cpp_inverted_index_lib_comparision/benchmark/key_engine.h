// "Phrase-as-key" engine: models a use case where a fixed-size tuple of
// terms (e.g. "vocA vocB vocC") is itself the atomic lookup key, as opposed
// to a classic inverted index where the same query would be decomposed into
// a boolean AND of individual terms. There is no tokenization, no posting
// list merge, no term dictionary -- just a single hash map from the full key
// string to the list of doc ids that contain that exact key.
//
// Semantics differ from a boolean AND query on the same terms: a doc only
// matches here if it contains the key's terms together as one of its keys
// (in that exact order), whereas `vocA AND vocB AND vocC` on a classic index
// matches any doc containing all three terms anywhere, even across unrelated
// keys. The key-map match set is therefore generally a subset of the
// AND-query match set.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class KeyEngine
{
public:
    // docKeys[docId] is the list of key strings (space-joined terms) that
    // doc docId contains.
    void build(const std::vector<std::vector<std::string>>& docKeys)
    {
        numDocs_ = static_cast<int>(docKeys.size());
        for (int docId = 0; docId < numDocs_; docId++)
        {
            for (const std::string& key : docKeys[docId])
            {
                std::vector<int>& postings = index_[key];
                if (postings.empty() || postings.back() != docId)
                    postings.push_back(docId);
            }
        }
    }

    // O(1) average-case hash lookup -- no merging, no scoring.
    std::vector<int> evaluate(const std::string& key) const
    {
        auto it = index_.find(key);
        return it == index_.end() ? std::vector<int> {} : it->second;
    }

    int numDocs() const { return numDocs_; }

private:
    std::unordered_map<std::string, std::vector<int>> index_;
    int                                                numDocs_ = 0;
};
