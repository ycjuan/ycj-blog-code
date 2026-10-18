// Synthetic corpus generator, parameterized by:
//   num_docs             - number of documents to generate
//   num_char_per_vocab   - length in characters of each distinct vocabulary term
//   phrase_card          - number of distinct *seed phrases* generated up front
//   num_vocab_per_phrase - number of terms making up each seed phrase
//   num_phrases_per_doc  - number of seed phrases sampled (with replacement)
//                          into each doc
//
// A fixed pool of `phrase_card` seed phrases is generated first (each a
// fixed-size tuple of `num_vocab_per_phrase` random terms). Every doc is then
// built by sampling `num_phrases_per_doc` of those *same* seed phrases, so
// a given phrase is reused across many docs whenever num_docs is large
// relative to phrase_card -- this is what makes the phrase-as-key benchmark
// meaningful (a key needs to actually recur across the corpus to have
// anything interesting to look up).
//
// This single corpus feeds both engines compared by the phrase-as-key
// benchmark:
//   - PISA/Xapian see each doc as the flat bag of all terms across its
//     sampled phrases (`Corpus::docs`), used to answer the classic boolean
//     AND query equivalent to a phrase lookup.
//   - KeyEngine sees each doc as its list of sampled phrase strings
//     (`Corpus::phrasesPerDoc`), looking up a whole phrase as one key.
#pragma once

#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

struct CorpusConfig
{
    int      numDocs           = 20000;
    int      numCharPerVocab   = 6;
    int      phraseCard        = 2000; // # of distinct seed phrases
    int      numVocabPerPhrase = 3;    // terms per seed phrase
    int      numPhrasesPerDoc  = 5;    // phrases sampled into each doc
    unsigned seed              = 42;
};

struct Corpus
{
    std::vector<std::string>              docs;          // doc_id -> space-joined terms of all its phrases
    std::vector<std::vector<std::string>> phrasesPerDoc; // doc_id -> list of phrase strings sampled for that doc
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

    Corpus corpus;

    // Generate phrase_card distinct seed phrases, each num_vocab_per_phrase
    // terms long.
    std::vector<std::string>        seedPhrases;
    std::unordered_set<std::string> seenPhrases;
    seedPhrases.reserve(cfg.phraseCard);
    while (static_cast<int>(seedPhrases.size()) < cfg.phraseCard)
    {
        std::ostringstream phraseOss;
        for (int i = 0; i < cfg.numVocabPerPhrase; i++)
        {
            std::string term = randomTerm();
            if (i > 0)
                phraseOss << ' ';
            phraseOss << term;
        }
        std::string phrase = phraseOss.str();
        if (seenPhrases.insert(phrase).second)
            seedPhrases.push_back(std::move(phrase));
    }

    // Build each doc by sampling num_phrases_per_doc of the seed phrases
    // (with replacement) -- the same fixed pool is reused across all docs.
    std::uniform_int_distribution<int> phraseDist(0, cfg.phraseCard - 1);
    corpus.docs.reserve(cfg.numDocs);
    corpus.phrasesPerDoc.reserve(cfg.numDocs);
    for (int docId = 0; docId < cfg.numDocs; docId++)
    {
        std::ostringstream       fullText;
        std::vector<std::string> phrases;
        phrases.reserve(cfg.numPhrasesPerDoc);
        for (int k = 0; k < cfg.numPhrasesPerDoc; k++)
        {
            const std::string& phrase = seedPhrases[phraseDist(rng)];
            if (fullText.tellp() > 0)
                fullText << ' ';
            fullText << phrase;
            phrases.push_back(phrase);
        }
        corpus.docs.push_back(fullText.str());
        corpus.phrasesPerDoc.push_back(std::move(phrases));
    }

    return corpus;
}
