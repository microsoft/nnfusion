#include "rang.hpp"
#include <string>

using namespace std;
using namespace rang;

int main()
{
    cout << endl
         << style::reset << bg::green << fg::gray << style::bold
         << " Rang works! " << bg::reset << fg::reset << style::reset << '\n'
         << endl;
}
