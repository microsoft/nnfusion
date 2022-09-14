// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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

LanguageUnit::LanguageUnit(const string symbol,
                           const string code,
                           const string header,
                           const string source)
    : symbol(symbol)
    , header_code(header)
    , source_code(source)
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
        NNFUSION_LOG(NNFUSION_WARNING) << "Empty required string.";
    }
    this->required.insert(required);
    return true;
}

bool LanguageUnit::require(shared_ptr<LanguageUnit> lu)
{
    NNFUSION_CHECK_NOT_NULLPTR(lu);
    if (!require(lu->get_symbol()))
        return false;
    this->local_symbol.emplace(lu->get_symbol(), lu);
    return true;
}

bool LanguageUnit::remove(shared_ptr<LanguageUnit> lu)
{
    NNFUSION_CHECK_NOT_NULLPTR(lu);
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
        NNFUSION_CHECK(this->local_symbol.find(it) != this->local_symbol.end())
            << "Cannot collect code from non-existed Language Unint.";
        NNFUSION_CHECK_NOT_NULLPTR(this->local_symbol[it])
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
        NNFUSION_CHECK(this->local_symbol.find(it) != this->local_symbol.end())
            << "Cannot collect code from non-existed Language Unint.";
        NNFUSION_CHECK_NOT_NULLPTR(this->local_symbol[it])
            << "Cannot collect code from non-existed null pointer.";
        lu << this->local_symbol[it]->collect_code() << "\n";
    }
    return lu.get_code();
}

void LanguageUnit::execute(bool append)
{
    auto cd = get_current_dir_name();
    NNFUSION_CHECK(setpwd());
    NNFUSION_CHECK(read_from_file());
    NNFUSION_CHECK(write_to_file(append));
    NNFUSION_CHECK(copy_all());
    if (this->pwd.empty())
        return;
    int status = chdir(cd);
    NNFUSION_CHECK(status == 0);
    return;
}

bool LanguageUnit::setpwd()
{
    if (this->pwd.empty())
        return true;

    size_t pos = pwd.find("/");
    while (pos != std::string::npos)
    {
        std::string dir = pwd.substr(0, pos);
        NNFUSION_CHECK(nnfusion::codegen::create_folder(dir));
        pos = pwd.find("/", pos + 1);
    }
    int status = chdir(pwd.c_str());
    return (status == 0);
}

bool LanguageUnit::write_to_file(bool append)
{
    if (this->write_to.empty())
        return true;

    if (this->write_to.substr(0, 6) == "shared")
    {
        if (source_code.empty() && header_code.empty())
        {
            divide_code();
        }

        std::ofstream header_file;
        std::ofstream source_file;
        if (append)
        {
            header_file.open("shared.h", std::ios::app);
            source_file.open(this->write_to, std::ios::app);
        }
        else
        {
            header_file.open("shared.h");
            source_file.open(this->write_to);
        }

        header_file << header_code;
        header_file.close();
        source_file << source_code;
        source_file.close();
    }
    else
    {
        std::ofstream file;
        if (append)
            file.open(this->write_to, std::ios::app);
        else
            file.open(this->write_to);
        file << this->get_code();
        file.close();
    }
    return true;
}

bool LanguageUnit::read_from_file()
{
    if (this->read_from.empty())
        return true;

    std::ifstream file(this->read_from, std::ios::out);
    if (!file.is_open())
        return false;
    std::string line;
    auto& myself = *this;
    while (std::getline(file, line))
        myself << line;
    file.close();
    return true;
}

bool LanguageUnit::copy_all()
{
    struct stat buffer;
    for (auto pair : copy_templates)
    {
        auto src_path = pair.first;
        auto tar_path = pair.second;
        if (stat(tar_path.c_str(), &buffer) == 0)
            continue;
        nnfusion::codegen::copy_file_from_templates(src_path, tar_path);
    }

    for (auto src_path : copy_folder)
    {
        int pos = src_path.find_last_of("/");
        auto tar_path = src_path.substr(pos + 1);
        if (stat(tar_path.c_str(), &buffer) == 0)
            continue;
        std::string cmd = std::string("cp -R ") + src_path + std::string(" .");
        NNFUSION_CHECK(system(cmd.c_str()) == 0) << "Failed to copy " << src_path;
    }
    return true;
}

void LanguageUnit::divide_code()
{
    auto remove_comment = [&](const std::string& code) {
        std::string res;
        bool s_cmt = false;
        bool m_cmt = false;
        for (int i = 0; i < code.length(); i++)
        {
            if (s_cmt == true && code[i] == '\n')
                s_cmt = false;

            else if (m_cmt == true && code[i] == '*' && code[i + 1] == '/')
                m_cmt = false, i++;

            else if (s_cmt || m_cmt)
                continue;

            else if (code[i] == '/' && code[i + 1] == '/')
                s_cmt = true, i++;
            else if (code[i] == '/' && code[i + 1] == '*')
                m_cmt = true, i++;

            else
                res += code[i];
        }
        return res;
    };

    auto process_header_and_macro = [&](const std::string& code) {
        std::string res;
        bool add_def_macro = false;
        bool add_if_macro = false;
        bool add_include = false;
        bool add_typedef = false;
        bool add_namespace = false;
        int i = 0;
        while (i < code.length())
        {
            if (add_if_macro)
            {
                if (code.substr(i, 6) == "#endif")
                {
                    add_if_macro = false;
                    header_code += "#endif\n";
                    i += 6;
                }
                else
                {
                    header_code += code[i];
                    i += 1;
                }
            }
            else if (add_def_macro)
            {
                header_code += code[i];
                if (code[i] == '\n' && i > 0 && code[i - 1] != '\\')
                {
                    add_def_macro = false;
                    header_code += "\n";
                }
                i += 1;
            }
            else if (add_include)
            {
                header_code += code[i];
                if (code[i] == '\n')
                {
                    add_include = false;
                }
                i += 1;
            }
            else if (add_typedef)
            {
                header_code += code[i];
                if (code[i] == '\n')
                {
                    add_typedef = false;
                }
                i += 1;
            }
            else if (add_namespace)
            {
                header_code += code[i];
                if (code[i] == '\n')
                {
                    add_namespace = false;
                }
                i += 1;
            }
            else if (code.substr(i, 3) == "#if")
            {
                header_code += "#if";
                add_if_macro = true;
                i += 3;
            }
            else if (code.substr(i, 7) == "#define")
            {
                header_code += "#define";
                add_def_macro = true;
                i += 7;
            }
            else if (code.substr(i, 9) == "#include ")
            {
                header_code += "#include ";
                add_include = true;
                i += 9;
            }
            else if (code.substr(i, 7) == "typedef")
            {
                header_code += "typedef";
                add_typedef = true;
                i += 7;
            }
            else if (code.substr(i, 16) == "using namespace ")
            {
                header_code += "using namespace ";
                add_namespace = true;
                i += 16;
            }
            else
            {
                res += code[i];
                i += 1;
            }
        }
        return res;
    };

    auto collect_functions_and_variables = [&](const std::string& code) {
        std::string cand;
        std::vector<std::string> blocks;
        bool add_block = false;
        int count = 0;
        int i = 0;
        while (i < code.length())
        {
            char ch = code[i];
            i += 1;
            cand += ch;
            if (add_block)
            {
                if (ch == '{')
                    count += 1;
                else if (ch == '}')
                    count -= 1;
                if (count == 0)
                {
                    if (i < code.length() && code[i] == ';')
                    {
                        cand += ';';
                        i += 1;
                    }
                    blocks.push_back(cand);
                    cand = "";
                    add_block = false;
                }
            }
            else if (ch == '{')
            {
                count = 1;
                add_block = true;
            }
        }

        if (cand.length() > 0)
            blocks.push_back(cand);

        std::vector<std::string> variables, functions;
        for (auto b : blocks)
        {
            int pos_f = b.find("{");
            if (pos_f >= 0)
            {
                int begin = 0;
                for (int i = 0; i < pos_f; i++)
                {
                    if (b[i] == ';')
                    {
                        std::string str = b.substr(begin, i + 1 - begin);
                        begin = i + 1;
                        variables.push_back(str);
                    }
                }

                std::string str = b.substr(begin, pos_f - begin);
                if (!str.empty())
                    str.erase(str.find_last_not_of(" ") + 1);
                if (!str.empty())
                {
                    if (str.back() == '=')
                        variables.push_back(b.substr(begin));
                    else
                        functions.push_back(b.substr(begin));
                }
            }
            else if (b.find(";") != string::npos)
                variables.push_back(b);
        }

        return std::make_pair(functions, variables);
    };

    auto process_functions = [&](std::vector<std::string>& functions) {
        for (auto f : functions)
        {
            std::string sig;
            int pos = f.find("{");
            if (pos >= 0)
            {
                sig = f.substr(0, pos);
                sig.erase(sig.find_last_not_of(" ") + 1);
                if (sig.find("inline") != string::npos || sig.find("template ") != string::npos ||
                    sig.find("static ") != string::npos)
                {
                    header_code += f + "\n";
                }
                else if (sig.find("extern ") != string::npos)
                {
                    header_code += sig + ";\n";
                    source_code += f;
                }
                else
                {
                    for (int i = 0; i < sig.length(); i++)
                    {
                        if (sig[i] != ' ' && sig[i] != '\n' && sig[i] != '\r' && sig[i] != '\t')
                        {
                            header_code += "\nextern " + sig.substr(i) + ";\n";
                            source_code += f;
                            break;
                        }
                    }
                }
            }
        }
    };

    auto process_variables = [&](std::vector<std::string>& variables) {
        for (auto v : variables)
        {
            if (v.find("extern ") != string::npos)
            {
                header_code += v;
                continue;
            }

            int pos_e = v.find("=");
            if (pos_e >= 0)
            {
                std::string decl = v.substr(0, pos_e);
                if (!decl.empty())
                    decl.erase(decl.find_last_not_of(" ") + 1);
                for (int i = 0; i < decl.length(); i++)
                {
                    if (decl[i] != ' ' && decl[i] != '\n' && decl[i] != '\r' && decl[i] != '\t')
                    {
                        header_code += "\nextern " + decl.substr(i) + ";\n";
                        source_code += v;
                        break;
                    }
                }
            }
            else
            {
                for (int i = 0; i < v.length(); i++)
                {
                    if (v[i] != ' ' && v[i] != '\n' && v[i] != '\r' && v[i] != '\t')
                    {
                        header_code += "\nextern " + v.substr(i);
                        source_code += v;
                        break;
                    }
                }
            }
        }
    };

    std::string code = get_code();
    auto code_wo_cmt = remove_comment(code);
    auto buffer = process_header_and_macro(code_wo_cmt);
    auto pair = collect_functions_and_variables(buffer);
    process_functions(pair.first);
    process_variables(pair.second);
}
