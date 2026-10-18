// Synthetic corpus generator parameterized by:
//   num_docs            - number of documents to generate
//   num_vocab_per_doc   - number of (possibly repeated) terms sampled into each doc
//   num_char_per_vocab  - length in characters of each distinct vocabulary term
//   vocab_card          - vocabulary cardinality: how many distinct terms exist overall
//
// Each doc is built by sampling `num_vocab_per_doc` terms uniformly at random
// (with replacement) from a fixed vocabulary of `vocab_card` distinct terms,
// each `num_char_per_vocab` characters long. Generation is seeded for
// reproducibility across engines/runs.
#pragma once

#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

struct CorpusConfig
{
    int      numDocs         = 20000;
    int      numVocabPerDoc  = 20;
    int      numCharPerVocab = 6;
    int      vocabCard       = 2000;
    unsigned seed            = 42;
};

struct Corpus
{
    std::vector<std::string> docs;  // doc_id -> space-joined term string
    std::vector<std::string> vocab; // vocab_id -> term string (vocab[i].size() == numCharPerVocab)
};

inline Corpus generateCorpus(const CorpusConfig& cfg)
{
    std::mt19937                       rng(cfg.seed);
    std::uniform_int_distribution<int> charDist(0, 25); // a-z

    auto randomTerm = [&]()
    {
        std::string term(cfg.numCharPerVocab, 'a');
        for (char& c : term)
            c = static_cast<char>('a' + charDist(rng));
        return term;
    };

    // Build a vocabulary of vocab_card *distinct* terms.
    Corpus corpus;
    corpus.vocab.reserve(cfg.vocabCard);
    {
        std::unordered_set<std::string> seen;
        seen.reserve(cfg.vocabCard * 2);
        while (static_cast<int>(corpus.vocab.size()) < cfg.vocabCard)
        {
            std::string term = randomTerm();
            if (seen.insert(term).second)
                corpus.vocab.push_back(term);
        }
    }

    // Each doc samples num_vocab_per_doc terms (with replacement) from the vocabulary.
    std::uniform_int_distribution<int> vocabDist(0, cfg.vocabCard - 1);
    corpus.docs.reserve(cfg.numDocs);
    for (int docId = 0; docId < cfg.numDocs; docId++)
    {
        std::ostringstream oss;
        for (int i = 0; i < cfg.numVocabPerDoc; i++)
        {
            if (i > 0)
                oss << ' ';
            oss << corpus.vocab[vocabDist(rng)];
        }
        corpus.docs.push_back(oss.str());
    }

    return corpus;
}
