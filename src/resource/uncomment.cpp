// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>

#include "uncomment.hpp"

using namespace std;

// start 23,749,645 in 1,912 files

void skip_comment(istream& s)
{
}

string uncomment(const string& s)
{
    stringstream out;
    for (size_t i = 0; i < s.size(); i++)
    {
        char c = s[i];
        if (i < s.size() - 1 && c == '/' && s[i + 1] == '/')
        {
            while (i < s.size() && c != '\n')
            {
                c = s[++i];
            }
            out << "\n";
        }
        else
        {
            out << c;
        }
    }
    return out.str();
}
