import os
import sys
import shutil
import winreg
import argparse
import logging
import subprocess

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def find_vs_path():
    # something like r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Microsoft\VC\v160"
    version = ["2019", "2017", "2015"]
    license = ["Enterprise", "Professional", "Community"]
    default_path = r"C:\Program Files (x86)\Microsoft Visual Studio"
    for v in version:
        v_path = os.path.join(default_path, v)
        if not os.path.isdir(v_path):
            continue
        for l in license:
            l_path = os.path.join(v_path, l)
            if not os.path.isdir(l_path):
                continue
            logger.info(f"Find Visual Studio in {l_path}")
            return l_path
    return ""


def copy_to(src, dst):
    assert os.path.exists(src), f"File not found: {src}"
    if os.path.isfile(dst):
        os.remove(dst)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    if os.path.isfile(src):
        shutil.copyfile(src, dst)
    if os.path.isdir(src):
        shutil.copytree(src, dst)


def copy_to_output(output_dir, build_type, platform):
    os.makedirs(output_dir, exist_ok=True)
    nnf_xbox_dir = r".\nnf_xbox_example"
    hlsl_path = os.path.join(nnf_xbox_dir, "HLSL")
    const_path = os.path.join(nnf_xbox_dir, "Constant")
    para_info = os.path.join(nnf_xbox_dir, "para_info.json")
    nnf_exe = os.path.join(
        nnf_xbox_dir, platform if "x64" in platform else "", build_type, "nnf_xbox_example.exe")
    deps_dir = os.path.join(
        nnf_xbox_dir, platform if "x64" in platform else "", r"Layout\Image\Loose")
    deps = []
    for file_name in os.listdir(deps_dir):
        if file_name.endswith(".dll"):
            deps.append(file_name)

    runtime_dir = r".\runtime"
    nnf_lib = os.path.join(
        runtime_dir, platform if "x64" in platform else "", build_type, "nnfusion_rt.dll")

    if os.path.exists(hlsl_path):
        copy_to(hlsl_path, os.path.join(output_dir, "HLSL"))
    if os.path.exists(const_path):
        copy_to(const_path, os.path.join(output_dir, "Constant"))
    copy_to(para_info, os.path.join(output_dir, "para_info.json"))
    copy_to(nnf_exe, os.path.join(output_dir, "nnf_xbox_example.exe"))
    copy_to(nnf_lib, os.path.join(output_dir, "nnfusion_rt.dll"))
    for file_name in deps:
        copy_to(os.path.join(deps_dir, file_name),
                os.path.join(output_dir, file_name))


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vs_path", default="",
                        help="visual studio install path")
    parser.add_argument("-t", "--build_type", default="Release")
    parser.add_argument("-p", "--platform", default="Gaming.Xbox.Scarlett.x64")
    parser.add_argument("-o", "--output", default="./build")
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    vs_path = args.vs_path if args.vs_path != "" else find_vs_path()
    build_type = args.build_type
    platform = args.platform
    output = args.output

    assert vs_path != "", "please specify vs install path by -v"
    msbuild_exe = os.path.join(vs_path, r"MSBuild\Current\Bin\MSBuild.exe")
    assert os.path.isfile(
        msbuild_exe), f"MSBuild.exe not found in {msbuild_exe}"

    try:
        subprocess.check_output([msbuild_exe, r".\nnf_xbox_example\nnf_xbox_example.vcxproj",
                                 f"/property:Configuration={build_type}", f"/property:Platform={platform}"], stderr=subprocess.STDOUT, encoding="utf8")
    except subprocess.CalledProcessError as e:
        logger.error(e.output)
        sys.exit(1)

    copy_to_output(output, build_type, platform)
    logger.info(f"Build sucessfully, output dir: {output}")


if __name__ == '__main__':
    main()
