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
            m_curl = curl_easy_init();
            if (m_curl)
            {
                std::regex pattern("(\\d{1,3}(\\.\\d{1,3}){3}):(\\d+)");
                std::smatch match;
                if (std::regex_search(address, match, pattern))
                {
                    curl_easy_setopt(m_curl, CURLOPT_URL, match[1].str().c_str());
                    curl_easy_setopt(m_curl, CURLOPT_PORT, std::stoi(match[3].str()));
                }
                else
                {
                    NNFUSION_CHECK(false) << "Invalid address format: " << address
                                          << "expect: <ip>:<port>";
                }
            }
        }

        ~CurlRequest()
        {
            curl_easy_cleanup(m_curl);
            curl_slist_free_all(m_chunk);
        }

        void add_custom_header(const std::string& header)
        {
            m_chunk = curl_slist_append(m_chunk, header.c_str());
        }

        bool send_request(std::string& buffer)
        {
            if (m_curl)
            {
                // Set custom set of headers.
                curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, m_chunk);
                curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, CurlRequest::write_callback);
                curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, &buffer);

                CURLcode res;
                res = curl_easy_perform(m_curl);
                if (res != CURLE_OK)
                {
                    fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
                }
                else
                {
                    return true;
                }
            }

            return false;
        }

        static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp)
        {
            ((std::string*)userp)->append((char*)contents, size * nmemb);
            return size * nmemb;
        }

    private:
        CURL* m_curl;
        struct curl_slist* m_chunk;
    };
}
