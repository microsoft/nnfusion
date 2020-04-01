// Microsoft (c) 2019, Wenxiang Hu
#include "languageunit.hpp"

LanguageUnit::LanguageUnit(const string symbol)
    : symbol(symbol)
{
}

LanguageUnit::LanguageUnit(const string symbol, const string code)
    : symbol(symbol)
{
    (*this) << code;
}

bool LanguageUnit::change_symbol(const string symbol)
{
    this->symbol = symbol;
    return true;
}

void LanguageUnit::clean_require()
{
    required.clear();
    local_symbol.clear();
}

bool LanguageUnit::require(const string required)
{
    //Todo(wenxh): check if the required string meets the grammar
    if (required.size() == 0)
    {
        LOG(WARNING) << "Empty required string.";
    }
    this->required.insert(required);
    return true;
}

bool LanguageUnit::require(shared_ptr<LanguageUnit> lu)
{
    CHECK_NOT_NULLPTR(lu);
    if (!require(lu->get_symbol()))
        return false;
    this->local_symbol.emplace(lu->get_symbol(), lu);
    return true;
}

bool LanguageUnit::remove(shared_ptr<LanguageUnit> lu)
{
    CHECK_NOT_NULLPTR(lu);
    auto sym = lu->get_symbol();
    this->required.erase(sym);
    this->local_symbol.erase(sym);
    return true;
}

bool LanguageUnit::replace(shared_ptr<LanguageUnit> lu, shared_ptr<LanguageUnit> b)
{
    auto sym = lu->get_symbol();
    if (required.count(sym) != 0)
    {
        this->required.erase(sym);
        require(b);
    }
    if (this->local_symbol.count(sym) != 0)
    {
        this->local_symbol.erase(sym);
        require(b);
    }
    return true;
}

string LanguageUnit::collect_code()
{
    LanguageUnit lu;
    for (auto& it : this->required)
    {
        CHECK(this->local_symbol.find(it) != this->local_symbol.end())
            << "Cannot collect code from non-existed Language Unint.";
        CHECK_NOT_NULLPTR(this->local_symbol[it])
            << "Cannot collect code from non-existed null pointer.";
        lu << this->local_symbol[it]->collect_code() << "\n";
    }
    lu << "// symbol: " << this->symbol << "\n";
    auto str = this->get_code();
    if (str.empty())
        lu << "// Empty Code\n";
    else
        lu << str;
    return lu.get_code();
}

string LanguageUnit::collect_required_code()
{
    LanguageUnit lu;
    for (auto& it : this->required)
    {
        CHECK(this->local_symbol.find(it) != this->local_symbol.end())
            << "Cannot collect code from non-existed Language Unint.";
        CHECK_NOT_NULLPTR(this->local_symbol[it])
            << "Cannot collect code from non-existed null pointer.";
        lu << this->local_symbol[it]->collect_code() << "\n";
    }
    return lu.get_code();
}