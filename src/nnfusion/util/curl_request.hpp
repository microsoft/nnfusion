// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <curl/curl.h>
#include <regex>
#include <string>
#include "nnfusion/util/errors.hpp"

namespace nnfusion
{
    class CurlRequest
    {
    public:
        CurlRequest(const std::string& address)
            : m_chunk(NULL)
        {
            m_curl_address = address;
            std::regex pattern("(\\d{1,3}(\\.\\d{1,3}){3}):(\\d+)");
            std::smatch match;
            if (!std::regex_search(address, match, pattern))
            {
                NNFUSION_CHECK(false) << "Invalid address format: " << address
                                        << "expect: <ip>:<port>";
            }
        }

        void add_custom_header(const std::string& header)
        {
            m_custom_headers.emplace_back(header);
        }

        bool send_request(std::string& buffer)
        {
            if (!m_curl_address.empty())
            {
                // Set custom headers.
                std::string headers_str = "";
                for (auto& header : m_custom_headers) {
                    headers_str += " -H \"" + header + "\"";
                }
                std::string cmd = "curl" + headers_str + " -d " + buffer + " -w " m_curl_address + " 2> /dev/null";

                static char line[80];
                std::string response;
                FILE* fp = popen(cmd.c_str(), "r");
                while (fgets(line, sizeof(line), fp))
                    response += line;
                pclose(fp);

                if (!response.empty())
                {
                    fprintf(stderr, "curl failed: %s\n", response.c_str());
                }
                else
                {
                    return true;
                }
            }

            return false;
        }

    private:
        std::string m_curl_address;
        std::vector<string> m_custom_headers;
    };
}
