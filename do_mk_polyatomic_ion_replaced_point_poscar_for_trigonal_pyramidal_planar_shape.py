import argparse
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mk_polyatomic_ion_replaced_point_poscar_for_trigonal_pyramidal_planar_shape import mk_polyatomic_ion_replaced_point_poscar


def wrap_mk_polyatomic_ion_replaced_point_poscar(args):
    return mk_polyatomic_ion_replaced_point_poscar(*args)


def mk_job_args(ion_contained_poscar_folder_p_list, target_ion_name, central_atom_symbol, neighboring_atom_symbol, bond_length_lower_end, bond_length_upper_end):
    # ターゲットとなるイオンの元素種を含むPOSCARとPOSCAR.nnlistのディレクトリパス一覧を取得
    poscar_add_atr = '/POSCAR'
    nnlist_add_str = '/nnlist_5/POSCAR.nnlist'
    gen_poscar_add_str = f'/{target_ion_name}_ion_replaced_point/POSCAR'
    ion_contained_poscar_path_list = [Path(str(p) + poscar_add_atr) for p in ion_contained_poscar_folder_p_list]
    ion_contained_nnlist_path_list = [Path(str(p) + nnlist_add_str) for p in ion_contained_poscar_folder_p_list]
    generated_poscar_path_list = [Path(str(p) + gen_poscar_add_str) for p in ion_contained_poscar_folder_p_list]
    number_of_poscar = len(ion_contained_poscar_folder_p_list)
    central_atom_symbol_list = [central_atom_symbol for i in range(number_of_poscar)]
    neighboring_atom_symbol_list = [neighboring_atom_symbol for i in range(number_of_poscar)]
    bond_length_lower_end_list = [bond_length_lower_end for i in range(number_of_poscar)]
    bond_length_upper_end_list = [bond_length_upper_end for i in range(number_of_poscar)]
    job_args = zip(ion_contained_poscar_path_list,
                   ion_contained_nnlist_path_list,
                   central_atom_symbol_list,
                   neighboring_atom_symbol_list,
                   bond_length_lower_end_list,
                   bond_length_upper_end_list,
                   generated_poscar_path_list)

    return job_args


# コマンドライン引数を受け取る
parser = argparse.ArgumentParser(description='This script takes five arguments: arg1, arg2, arg3, arg4, arg5 and arg6.',
                                 usage='%(prog)s <arg1> <arg2> <arg3> <arg4> <arg5> <arg6> \
                                 \nexample: python3 %(prog)s CO3 C O 0.99 1.66 ../get_some_ion_contained_pos_folder_p_list/CO3_contained_poscar_folder_path_list_ver2.npy')
parser.add_argument('arg1', help='target_ion_name: CO3')
parser.add_argument('arg2', help='central_atom_symbol: C')
parser.add_argument('arg3', help='neighboring_atom_symbol: O')
parser.add_argument('arg4', help='bond_length_lower_end: 0.99')
parser.add_argument('arg5', help='bond_length_upper_end: 1.66')
parser.add_argument('arg6', help='npy_file_path: ../get_some_ion_contained_pos_folder_p_list/CO3_contained_poscar_folder_path_list_ver2.npy')
args = parser.parse_args()
target_ion_name = args.arg1
central_atom_symbol = args.arg2
neighboring_atom_symbol = args.arg3
bond_length_lower_end = args.arg4
bond_length_upper_end = args.arg5
target_npy_p = args.arg6
print(f'target_ion_name: {target_ion_name}')
print(f'central_atom_symbol: {central_atom_symbol}')
print(f'neighboring_atom_symbol: {neighboring_atom_symbol}')
print(f'bond_length_lower_end: {bond_length_lower_end}')
print(f'bond_length_upper_end: {bond_length_upper_end}')
print(f'target_npy_p: {target_npy_p}')
print(f'os.path.exists(target_npy_p): {os.path.exists(target_npy_p)}')

# 炭酸イオンを含む結晶構造ファイルパス一覧
ion_contained_poscar_folder_p_list = np.load(target_npy_p, allow_pickle=True)

job_args = mk_job_args(ion_contained_poscar_folder_p_list,
                       target_ion_name=target_ion_name,
                       central_atom_symbol=central_atom_symbol,
                       neighboring_atom_symbol=neighboring_atom_symbol,
                       bond_length_lower_end=bond_length_lower_end,
                       bond_length_upper_end=bond_length_upper_end)

# 並列化
pp = Pool(cpu_count() - 1)
total = len(ion_contained_poscar_folder_p_list)
try:
    func_results = list(tqdm(pp.imap(wrap_mk_polyatomic_ion_replaced_point_poscar, job_args), total=total))
finally:
    pp.close()
    pp.join()
