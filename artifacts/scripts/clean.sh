declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $THIS_SCRIPT_DIR/..

echo "clean figure10"
cd figure10/
rm -rf logs rammer rammer_base | true
rm reproduce_result/*.dat | true
cd ..

echo "clean figure11"
cd figure11/
rm -rf logs rammer rammer_base | true
rm reproduce_result/*.dat | true
cd ..

echo "clean figure12"
cd figure12/
rm -rf logs rammer rammer_base | true
rm reproduce_result/*.dat | true
cd ..

echo "clean figure14"
cd figure14/
rm -rf logs rammer rammer_base | true
rm reproduce_result/*.dat | true
cd ..

echo "clean figure15"
cd figure15/
rm -rf logs rammer_fast rammer_base_fast rammer_select rammer_base_select | true
rm reproduce_result/*.dat | true
cd ..
