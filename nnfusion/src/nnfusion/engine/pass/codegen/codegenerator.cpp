// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "codegenerator.hpp"
#include <queue>
#include "nnfusion/core/kernels/common_langunit.hpp"

DECLARE_bool(fkernels_as_files);
DECLARE_int64(fkernels_files_number);

using namespace nnfusion;
using namespace nnfusion::codegen;

void LanguageUnitwithVec::collect_requirement()
{
    for (auto unit : unit_vec)
    {
        for (auto& it : unit->local_symbol)
        {
            this->require(it.second);
        }
        unit->clean_require();
    }
    return;
}

void LanguageUnitwithVec::execute(bool append)
{
    // validation check
    if (!this->pwd.empty() || !this->write_to.empty() || !this->copy_templates.empty() ||
        !this->read_from.empty())
    {
        for (auto lu : unit_vec)
        {
            if (!this->pwd.empty() && lu->pwd.empty())
                lu->pwd = this->pwd;
            NNFUSION_CHECK(this->pwd == lu->pwd) << "Conflict pwd: " << this->pwd << " vs "
                                                 << lu->pwd;

            if (!this->write_to.empty() && lu->write_to.empty())
                lu->write_to = this->write_to;
            NNFUSION_CHECK(this->write_to == lu->write_to)
                << "Conflict write_to: " << this->write_to << " vs " << lu->write_to;

            if (!this->read_from.empty() && lu->read_from.empty())
                lu->read_from = this->read_from;
            NNFUSION_CHECK(this->read_from == lu->read_from)
                << "Conflict read_from: " << this->read_from << " vs " << lu->read_from;

            if (!this->copy_templates.empty() && lu->copy_templates.empty())
                lu->copy_templates = this->copy_templates;
            NNFUSION_CHECK(this->copy_templates == lu->copy_templates)
                << "Conflict copy_templates.";
        }
    }
    // execution
    for (auto lu : unit_vec)
        lu->execute(append);
    return;
}

LanguageUnit_p
    CodegenFuncCallsUnit::wrap(LanguageUnit_p new_caller, LanguageUnit_p begin, LanguageUnit_p end)
{
    CodegenMainBlockUnit_p block =
        std::make_shared<CodegenMainBlockUnit>(this->symbol + "_wrap_block");
    block->unit_vec = this->unit_vec;
    block->begin = begin;
    block->end = end;
    block->local_symbol = this->local_symbol;
    new_caller->require(block);
    return new_caller;
}

void CodegenMainBlockUnit::execute(bool append)
{
    // validation check
    std::vector<LanguageUnit_p> all_unit;
    all_unit.push_back(begin);
    all_unit.insert(all_unit.end(), unit_vec.begin(), unit_vec.end());
    all_unit.push_back(end);

    if (!this->pwd.empty() || !this->write_to.empty() || !this->copy_templates.empty() ||
        !this->read_from.empty())
    {
        for (auto lu : all_unit)
        {
            if (!this->pwd.empty() && lu->pwd.empty())
                lu->pwd = this->pwd;
            NNFUSION_CHECK(this->pwd == lu->pwd) << "Conflict pwd: " << this->pwd << " vs "
                                                 << lu->pwd;

            if (!this->write_to.empty() && lu->write_to.empty())
                lu->write_to = this->write_to;
            NNFUSION_CHECK(this->write_to == lu->write_to)
                << "Conflict write_to: " << this->write_to << " vs " << lu->write_to;

            if (!this->read_from.empty() && lu->read_from.empty())
                lu->read_from = this->read_from;
            NNFUSION_CHECK(this->read_from == lu->read_from)
                << "Conflict read_from: " << this->read_from << " vs " << lu->read_from;

            if (!this->copy_templates.empty() && lu->copy_templates.empty())
                lu->copy_templates = this->copy_templates;
            NNFUSION_CHECK(this->copy_templates == lu->copy_templates)
                << "Conflict copy_templates.";
        }
    }

    // execution
    for (auto lu : all_unit)
        lu->execute(append);

    all_unit.clear();

    return;
}

void CodegenMainBlockUnit::collect_requirement()
{
    std::vector<LanguageUnit_p> all_unit;
    all_unit.push_back(begin);
    all_unit.insert(all_unit.end(), unit_vec.begin(), unit_vec.end());
    all_unit.push_back(end);

    for (auto unit : all_unit)
    {
        for (auto& it : unit->local_symbol)
        {
            this->require(it.second);
        }
        unit->clean_require();
    }

    all_unit.clear();

    return;
}

bool CodeGenerator::codegen()
{
    // pass exec info: pwd, write_to
    pass_exec_info();

    // sort LanguageUnit
    std::vector<LanguageUnit_p> sorted_unit;

    std::unordered_set<LanguageUnit_p> visited;
    std::unordered_map<LanguageUnit_p, std::vector<LanguageUnit_p>> required_by;
    std::unordered_map<LanguageUnit_p, int> ind;

    auto comparef = [](const LanguageUnit_p& a, const LanguageUnit_p& b) {
        auto prior = [](LanguageUnit_p lup) {
            if (lup->symbol.find("header::") != string::npos)
                return 0;
            if (lup->symbol.find("macro::") != string::npos)
                return 1;
            if (lup->symbol.find("declaration::") != string::npos)
                return 2;
            return 3;
        };
        return prior(a) > prior(b);
    };

    std::priority_queue<LanguageUnit_p, std::vector<LanguageUnit_p>, decltype(comparef)>
        prior_queue(comparef);
    std::queue<LanguageUnit_p> queue;
    queue.push(lup_codegen);
    while (!queue.empty())
    {
        auto cur = queue.front();
        queue.pop();
        if (visited.find(cur) != visited.end())
            continue;
        visited.insert(cur);
        ind[cur] = cur->local_symbol.size();
        if (ind[cur] == 0)
        {
            prior_queue.push(cur);
            continue;
        }
        for (auto& it : cur->local_symbol)
        {
            LanguageUnit_p lu = it.second;
            required_by[lu].push_back(cur);
            queue.push(lu);
        }
    }

    while (!prior_queue.empty())
    {
        auto cur = prior_queue.top();
        prior_queue.pop();
        sorted_unit.push_back(cur);
        for (auto it : required_by[cur])
        {
            ind[it] -= 1;

            if (ind[it] == 0)
            {
                prior_queue.push(it);
            }
        }
    }

    std::unordered_set<std::string> codegen_files;
    auto clear_file = [&](const std::string& pwd, const std::string& write_to) {
        int pos = pwd.find("/");
        while (pos != std::string::npos)
        {
            std::string dir = pwd.substr(0, pos);
            NNFUSION_CHECK(nnfusion::codegen::create_folder(dir));
            pos = pwd.find("/", pos + 1);
        }

        std::string search;
        if (write_to.empty() || write_to[0] == '/')
            search = write_to;
        else
            search = pwd;

        if (!pwd.empty() && pwd.back() != '/')
            search = pwd + "/";

        std::string shared_header;
        if (write_to.substr(0, 6) == "shared")
        {
            shared_header = search + "shared.h";
            if (codegen_files.find(shared_header) == codegen_files.end())
            {
                codegen_files.insert(shared_header);
                std::ofstream file;
                file.open(shared_header);
                // file << nnfusion::kernels::boilerplate::MIT1->get_code();
                file.close();
                // struct stat buffer;
                // if (stat(shared_header.c_str(), &buffer) == 0)
                // {
                //     NNFUSION_CHECK(remove(shared_header.c_str()) == 0);
                // }
            }
        }

        search = pwd + write_to;
        if (codegen_files.find(search) == codegen_files.end())
        {
            codegen_files.insert(search);
            std::ofstream file;
            file.open(search);
            // if (search.find(".txt", search.size() - 4) == string::npos)
            // {
            //     file << nnfusion::kernels::boilerplate::MIT1->get_code();
            // }
            // else
            // {
            //     file << nnfusion::kernels::boilerplate::MIT2->get_code();
            // }

            if (files_include_shared.find(search) != files_include_shared.end())
            {
                file << "#include \"shared.h\"\n";
            }

            file.close();
            // struct stat buffer;
            // if (stat(search.c_str(), &buffer) == 0)
            // {
            //     NNFUSION_CHECK(remove(search.c_str()) == 0);
            // }
        }
    };

    // write code
    std::unordered_set<std::string> executed;
    for (auto lu : sorted_unit)
    {
        clear_file(lu->pwd, lu->write_to);
        std::string search_name = lu->symbol + "_" + lu->pwd + "_" + lu->write_to;
        if (executed.find(search_name) == executed.end())
        {
            // NNFUSION_LOG(INFO) << lu->symbol << "\t" << lu->pwd << "\t" << lu->write_to;
            lu->execute();
            executed.insert(search_name);
        }
    }

    for (auto search : codegen_files)
    {
        std::ofstream file;
        file.open(search, std::ios::app);
        if (search.find(".txt", search.size() - 4) == string::npos)
        {
            file << nnfusion::kernels::boilerplate::MIT1->get_code();
        }
        else
        {
            file << nnfusion::kernels::boilerplate::MIT2->get_code();
        }
        file.close();
    }

    return true;
}

void CodeGenerator::pass_exec_info()
{
    std::queue<LanguageUnit_p> queue;
    std::unordered_set<LanguageUnit_p> visited;
    queue.push(lup_codegen);
    std::unordered_map<LanguageUnit_p, LanguageUnit_p> receiver_giver;

    auto pass = [&](LanguageUnit_p giver, LanguageUnit_p receiver) {

        // NNFUSION_LOG(INFO) << "==================";
        // NNFUSION_LOG(INFO) << giver->symbol << "\t" << giver->pwd << "\t" << giver->write_to;
        // NNFUSION_LOG(INFO) << receiver->symbol << "\t" << receiver->pwd << "\t" << receiver->write_to;
        if (receiver_giver.find(receiver) != receiver_giver.end())
        {
            auto first_giver = receiver_giver[receiver];
            if ((!giver->pwd.empty() && giver->pwd != receiver->pwd) ||
                (!giver->write_to.empty() && giver->write_to != receiver->write_to))
            {
                receiver->pwd = get_codegen_folder();
                receiver->write_to = "shared" + m_kernel_suffix;
                files_include_shared.insert(receiver->pwd + receiver->write_to);
                files_include_shared.insert(giver->pwd + giver->write_to);
                files_include_shared.insert(first_giver->pwd + first_giver->write_to);
            }
            return;
        }

        bool passed = false;
        if (!giver->pwd.empty() && receiver->pwd.empty())
        {
            receiver->pwd = giver->pwd;
            passed = true;
        }

        if (!giver->write_to.empty() && receiver->write_to.empty())
        {
            receiver->write_to = giver->write_to;
            passed = true;
        }
        if (passed)
            receiver_giver[receiver] = giver;

        // NNFUSION_LOG(INFO) << "----------------";
        // NNFUSION_LOG(INFO) << giver->symbol << "\t" << giver->pwd << "\t" << giver->write_to;
        // NNFUSION_LOG(INFO) << receiver->symbol << "\t" << receiver->pwd << "\t" << receiver->write_to;
    };

    auto pass_to_sub = [&](LanguageUnitwithVec_p start) {
        std::queue<LanguageUnitwithVec_p> lu_with_vec;
        std::vector<LanguageUnitwithVec_p> all_lu_with_vec;
        std::queue<LanguageUnit_p> non_vec_sub_lu;
        lu_with_vec.push(start);
        all_lu_with_vec.push_back(start);

        while (!lu_with_vec.empty())
        {
            auto curr = lu_with_vec.front();
            lu_with_vec.pop();
            for (LanguageUnit_p sub_lu : curr->unit_vec)
            {
                pass(curr, sub_lu);
                if (auto sub_lu_is_vec = std::dynamic_pointer_cast<LanguageUnitwithVec>(sub_lu))
                {
                    lu_with_vec.push(sub_lu_is_vec);
                    all_lu_with_vec.push_back(sub_lu_is_vec);
                }
                else
                {
                    non_vec_sub_lu.push(sub_lu);
                }
            }
        }

        while (!non_vec_sub_lu.empty())
        {
            LanguageUnit_p curr = non_vec_sub_lu.front();
            non_vec_sub_lu.pop();
            if (visited.find(curr) != visited.end())
                continue;
            visited.insert(curr);
            for (auto& it : curr->local_symbol)
            {
                LanguageUnit_p sym = it.second;
                pass(curr, sym);
                non_vec_sub_lu.push(sym);
            }
        }

        std::reverse(all_lu_with_vec.begin(), all_lu_with_vec.end());
        for (auto curr : all_lu_with_vec)
        {
            for (auto& it : curr->local_symbol)
            {
                pass(curr, it.second);
            }
            curr->collect_requirement();
        }

    };

    while (!queue.empty())
    {
        LanguageUnit_p curr = queue.front();
        queue.pop();
        if (visited.find(curr) != visited.end())
            continue;
        visited.insert(curr);

        if (auto curr_is_vec = std::dynamic_pointer_cast<LanguageUnitwithVec>(curr))
        {
            pass_to_sub(curr_is_vec);
        }

        for (auto& it : curr->local_symbol)
        {
            LanguageUnit_p sym = it.second;
            pass(curr, sym);
            queue.push(sym);
        }
    }
}