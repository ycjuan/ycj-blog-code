// Generic boolean query AST + parser, shared between engines so both PISA-
// style and Xapian benchmarks evaluate *the exact same parsed query tree*,
// just via different evaluators (see pisa_engine.h / xapian_engine.h).
//
// Grammar (standard precedence: NOT > AND > OR), same as
// ../pisa/main.cpp's BooleanQueryEngine:
//   expr   := andExpr (OR andExpr)*
//   andExpr:= notExpr (AND notExpr)*
//   notExpr:= NOT notExpr | primary
//   primary:= TERM | '(' expr ')'
#pragma once

#include <cctype>
#include <memory>
#include <string>
#include <vector>

enum class NodeType
{
    Term,
    And,
    Or,
    Not,
};

struct QueryNode
{
    NodeType                   type;
    std::string                term;        // valid when type == Term
    std::unique_ptr<QueryNode> left, right; // right used only for Not (as `left`)

    static std::unique_ptr<QueryNode> makeTerm(std::string t)
    {
        auto node  = std::make_unique<QueryNode>();
        node->type = NodeType::Term;
        node->term = std::move(t);
        return node;
    }

    static std::unique_ptr<QueryNode> makeBinary(NodeType                   type,
                                                 std::unique_ptr<QueryNode> l,
                                                 std::unique_ptr<QueryNode> r)
    {
        auto node   = std::make_unique<QueryNode>();
        node->type  = type;
        node->left  = std::move(l);
        node->right = std::move(r);
        return node;
    }

    static std::unique_ptr<QueryNode> makeNot(std::unique_ptr<QueryNode> child)
    {
        auto node  = std::make_unique<QueryNode>();
        node->type = NodeType::Not;
        node->left = std::move(child);
        return node;
    }
};

class QueryParser
{
public:
    static std::unique_ptr<QueryNode> parse(const std::string& query)
    {
        QueryParser parser(tokenize(query));
        return parser.parseExpr();
    }

private:
    explicit QueryParser(std::vector<std::string> tokens)
        : tokens_(std::move(tokens))
    {
    }

    std::unique_ptr<QueryNode> parseExpr() // OR
    {
        std::unique_ptr<QueryNode> result = parseAnd();
        while (peek() == "OR")
        {
            advance();
            result = QueryNode::makeBinary(NodeType::Or, std::move(result), parseAnd());
        }
        return result;
    }

    std::unique_ptr<QueryNode> parseAnd()
    {
        std::unique_ptr<QueryNode> result = parseNot();
        while (peek() == "AND")
        {
            advance();
            result = QueryNode::makeBinary(NodeType::And, std::move(result), parseNot());
        }
        return result;
    }

    std::unique_ptr<QueryNode> parseNot()
    {
        if (peek() == "NOT")
        {
            advance();
            return QueryNode::makeNot(parseNot());
        }
        return parsePrimary();
    }

    std::unique_ptr<QueryNode> parsePrimary()
    {
        if (peek() == "(")
        {
            advance();
            std::unique_ptr<QueryNode> result = parseExpr();
            if (peek() == ")")
                advance();
            return result;
        }
        std::string term = peek();
        advance();
        return QueryNode::makeTerm(term);
    }

    const std::string& peek() const
    {
        static const std::string kEnd;
        return pos_ < tokens_.size() ? tokens_[pos_] : kEnd;
    }

    void advance() { pos_++; }

    static std::vector<std::string> tokenize(const std::string& query)
    {
        std::vector<std::string> tokens;
        std::string              current;
        auto                     flush = [&]()
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
                tokens.push_back(std::string(1, c));
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

    std::vector<std::string> tokens_;
    size_t                   pos_ = 0;
};
