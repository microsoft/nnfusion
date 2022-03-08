// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <regex>
#include <string>
#include "nnfusion/util/errors.hpp"

namespace nnfusion
{
    class CurlRequest
    {
    public:
        CurlRequest(const std::string& address)
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

        void add_custom_header(const std::string& header) { m_custom_headers.emplace_back(header); }
        bool send_request(std::string& buffer)
        {
            if (!m_curl_address.empty())
            {
                // Set custom headers.
                std::string headers_str = "";
                for (auto& header : m_custom_headers)
                {
                    headers_str += " -H \'" + header + "\'";
                }
                std::string cmd = "curl -X GET" + headers_str + " -i -d -w " + m_curl_address;

                static char line[4096];
                std::string response;
                FILE* fp = popen(cmd.c_str(), "r");
                while (fgets(line, sizeof(line), fp))
                    response += line;
                pclose(fp);

                if (response.substr(0, 12) != "HTTP/1.1 200")
                {
                    fprintf(stderr, "curl failed: %s\n", response.c_str());
                }
                else
                {
                    buffer = get_response_body(response);
                    return true;
                }
            }

            return false;
        }

    private:
        std::string m_curl_address;
        std::vector<string> m_custom_headers;

        string get_response_body(const std::string response)
        {
            const std::string end_of_header = "Transfer-Encoding: chunked";
            auto found = response.find(end_of_header);
            if (found != std::string::npos)
            {
                return response.substr(found + end_of_header.size() + 1);
            }
            return response;
        }
    };
}
