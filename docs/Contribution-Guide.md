This is the contribution guide for NNFusion project on Github.
## Before contributing your code:
Read [Build Guide](https://github.com/microsoft/nnfusion/blob/master/docs/Build-Guide.md) to know how to build your code in your dev machine.    
Read [Coding Guide](https://github.com/microsoft/nnfusion/blob/master/docs/Coding-Guide.md) to know how to modify source code.

## Their are two ways of contributing your code through pull requests:
If you are a member of NNFusion team, you can 
   1. Create a branch in the repo 
   2. Commit your code
   3. Run maint/script/apply_code_style.sh
   4. Run maint/script/test.sh
   5. Submit the PR with you update;   

And if your are not, you can also do
   1. Fork the NNFusion repo
   2. Create your branch in your forked repo
   3. Commit your code
   4. Run maint/script/apply_code_style.sh
   5. Run maint/script/test.sh
   6. Submit your PR with the branch in your repo
   7. Due to security reasons, your PR will not be checked automatically, thus you need to notify or wait a project member to comment "/AzurePipelines run" on your pull request

Here is an example to illustrate what the PR should looks like: [PR#10](https://github.com/microsoft/nnfusion/pull/10).

## Here are some rules your PR needs to comply:
   - [ ] At least ONE NNFusion team member exclude the committer approves the PR
   - [ ] The source code requires test cases to cover the changed part
   - [ ] The source code must have no license issue
   - [ ] The source code must pass all checks

## At last,
If all rules are followed, the NNFusion team member would help you to merge your pull request.
