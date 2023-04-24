import torch
import torch.nn as nn

class RAEUnroll(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1024, 512)
    
    def forward(self, inp):
        output_32 = inp[32]
        output_20 = inp[20]
        output_21 = inp[21]
        output_22 = inp[22]
        output_23 = inp[23]
        output_24 = inp[24]
        output_25 = inp[25]
        output_30 = inp[30]
        output_31 = inp[31]
        output_33 = self.encoder(torch.cat((output_30, output_31)))
        output_33 = torch.tanh(output_33)
        output_26 = inp[26]
        output_27 = inp[27]
        output_28 = inp[28]
        output_29 = inp[29]
        output_34 = self.encoder(torch.cat((output_28, output_29)))
        output_34 = torch.tanh(output_34)
        output_35 = self.encoder(torch.cat((output_27, output_34)))
        output_35 = torch.tanh(output_35)
        output_36 = self.encoder(torch.cat((output_26, output_35)))
        output_36 = torch.tanh(output_36)
        output_37 = self.encoder(torch.cat((output_33, output_36)))
        output_37 = torch.tanh(output_37)
        output_38 = self.encoder(torch.cat((output_25, output_37)))
        output_38 = torch.tanh(output_38)
        output_39 = self.encoder(torch.cat((output_24, output_38)))
        output_39 = torch.tanh(output_39)
        output_40 = self.encoder(torch.cat((output_23, output_39)))
        output_40 = torch.tanh(output_40)
        output_41 = self.encoder(torch.cat((output_22, output_40)))
        output_41 = torch.tanh(output_41)
        output_42 = self.encoder(torch.cat((output_21, output_41)))
        output_42 = torch.tanh(output_42)
        output_43 = self.encoder(torch.cat((output_20, output_42)))
        output_43 = torch.tanh(output_43)
        output_44 = self.encoder(torch.cat((output_32, output_43)))
        output_44 = torch.tanh(output_44)
        output_17 = inp[17]
        output_18 = inp[18]
        output_19 = inp[19]
        output_45 = self.encoder(torch.cat((output_18, output_19)))
        output_45 = torch.tanh(output_45)
        output_46 = self.encoder(torch.cat((output_17, output_45)))
        output_46 = torch.tanh(output_46)
        output_11 = inp[11]
        output_12 = inp[12]
        output_13 = inp[13]
        output_14 = inp[14]
        output_15 = inp[15]
        output_16 = inp[16]
        output_47 = self.encoder(torch.cat((output_15, output_16)))
        output_47 = torch.tanh(output_47)
        output_48 = self.encoder(torch.cat((output_14, output_47)))
        output_48 = torch.tanh(output_48)
        output_49 = self.encoder(torch.cat((output_13, output_48)))
        output_49 = torch.tanh(output_49)
        output_50 = self.encoder(torch.cat((output_12, output_49)))
        output_50 = torch.tanh(output_50)
        output_51 = self.encoder(torch.cat((output_11, output_50)))
        output_51 = torch.tanh(output_51)
        output_52 = self.encoder(torch.cat((output_46, output_51)))
        output_52 = torch.tanh(output_52)
        output_10 = inp[10]
        output_8 = inp[8]
        output_9 = inp[9]
        output_53 = self.encoder(torch.cat((output_8, output_9)))
        output_53 = torch.tanh(output_53)
        output_5 = inp[5]
        output_6 = inp[6]
        output_7 = inp[7]
        output_54 = self.encoder(torch.cat((output_6, output_7)))
        output_54 = torch.tanh(output_54)
        output_55 = self.encoder(torch.cat((output_5, output_54)))
        output_55 = torch.tanh(output_55)
        output_56 = self.encoder(torch.cat((output_53, output_55)))
        output_56 = torch.tanh(output_56)
        output_57 = self.encoder(torch.cat((output_10, output_56)))
        output_57 = torch.tanh(output_57)
        output_58 = self.encoder(torch.cat((output_52, output_57)))
        output_58 = torch.tanh(output_58)
        output_4 = inp[4]
        output_0 = inp[0]
        output_1 = inp[1]
        output_2 = inp[2]
        output_3 = inp[3]
        output_59 = self.encoder(torch.cat((output_2, output_3)))
        output_59 = torch.tanh(output_59)
        output_60 = self.encoder(torch.cat((output_1, output_59)))
        output_60 = torch.tanh(output_60)
        output_61 = self.encoder(torch.cat((output_0, output_60)))
        output_61 = torch.tanh(output_61)
        output_62 = self.encoder(torch.cat((output_4, output_61)))
        output_62 = torch.tanh(output_62)
        output_63 = self.encoder(torch.cat((output_58, output_62)))
        output_63 = torch.tanh(output_63)
        output_64 = self.encoder(torch.cat((output_44, output_63)))
        output_64 = torch.tanh(output_64)
        return output_64

# complete binary tree
'''
class RAEUnroll(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1024, 512)
    
    def forward(self, inp):
        output_63 = inp[63]
        output_64 = inp[64]
        output_31 = self.encoder(torch.cat((output_63, output_64)))
        output_31 = torch.tanh(output_31)
        output_65 = inp[65]
        output_66 = inp[66]
        output_32 = self.encoder(torch.cat((output_65, output_66)))
        output_32 = torch.tanh(output_32)
        output_15 = self.encoder(torch.cat((output_31, output_32)))
        output_15 = torch.tanh(output_15)
        output_67 = inp[67]
        output_68 = inp[68]
        output_33 = self.encoder(torch.cat((output_67, output_68)))
        output_33 = torch.tanh(output_33)
        output_69 = inp[69]
        output_70 = inp[70]
        output_34 = self.encoder(torch.cat((output_69, output_70)))
        output_34 = torch.tanh(output_34)
        output_16 = self.encoder(torch.cat((output_33, output_34)))
        output_16 = torch.tanh(output_16)
        output_7 = self.encoder(torch.cat((output_15, output_16)))
        output_7 = torch.tanh(output_7)
        output_71 = inp[71]
        output_72 = inp[72]
        output_35 = self.encoder(torch.cat((output_71, output_72)))
        output_35 = torch.tanh(output_35)
        output_73 = inp[73]
        output_74 = inp[74]
        output_36 = self.encoder(torch.cat((output_73, output_74)))
        output_36 = torch.tanh(output_36)
        output_17 = self.encoder(torch.cat((output_35, output_36)))
        output_17 = torch.tanh(output_17)
        output_75 = inp[75]
        output_76 = inp[76]
        output_37 = self.encoder(torch.cat((output_75, output_76)))
        output_37 = torch.tanh(output_37)
        output_77 = inp[77]
        output_78 = inp[78]
        output_38 = self.encoder(torch.cat((output_77, output_78)))
        output_38 = torch.tanh(output_38)
        output_18 = self.encoder(torch.cat((output_37, output_38)))
        output_18 = torch.tanh(output_18)
        output_8 = self.encoder(torch.cat((output_17, output_18)))
        output_8 = torch.tanh(output_8)
        output_3 = self.encoder(torch.cat((output_7, output_8)))
        output_3 = torch.tanh(output_3)
        output_79 = inp[79]
        output_80 = inp[80]
        output_39 = self.encoder(torch.cat((output_79, output_80)))
        output_39 = torch.tanh(output_39)
        output_81 = inp[81]
        output_82 = inp[82]
        output_40 = self.encoder(torch.cat((output_81, output_82)))
        output_40 = torch.tanh(output_40)
        output_19 = self.encoder(torch.cat((output_39, output_40)))
        output_19 = torch.tanh(output_19)
        output_83 = inp[83]
        output_84 = inp[84]
        output_41 = self.encoder(torch.cat((output_83, output_84)))
        output_41 = torch.tanh(output_41)
        output_85 = inp[85]
        output_86 = inp[86]
        output_42 = self.encoder(torch.cat((output_85, output_86)))
        output_42 = torch.tanh(output_42)
        output_20 = self.encoder(torch.cat((output_41, output_42)))
        output_20 = torch.tanh(output_20)
        output_9 = self.encoder(torch.cat((output_19, output_20)))
        output_9 = torch.tanh(output_9)
        output_87 = inp[87]
        output_88 = inp[88]
        output_43 = self.encoder(torch.cat((output_87, output_88)))
        output_43 = torch.tanh(output_43)
        output_89 = inp[89]
        output_90 = inp[90]
        output_44 = self.encoder(torch.cat((output_89, output_90)))
        output_44 = torch.tanh(output_44)
        output_21 = self.encoder(torch.cat((output_43, output_44)))
        output_21 = torch.tanh(output_21)
        output_91 = inp[91]
        output_92 = inp[92]
        output_45 = self.encoder(torch.cat((output_91, output_92)))
        output_45 = torch.tanh(output_45)
        output_93 = inp[93]
        output_94 = inp[94]
        output_46 = self.encoder(torch.cat((output_93, output_94)))
        output_46 = torch.tanh(output_46)
        output_22 = self.encoder(torch.cat((output_45, output_46)))
        output_22 = torch.tanh(output_22)
        output_10 = self.encoder(torch.cat((output_21, output_22)))
        output_10 = torch.tanh(output_10)
        output_4 = self.encoder(torch.cat((output_9, output_10)))
        output_4 = torch.tanh(output_4)
        output_1 = self.encoder(torch.cat((output_3, output_4)))
        output_1 = torch.tanh(output_1)
        output_95 = inp[95]
        output_96 = inp[96]
        output_47 = self.encoder(torch.cat((output_95, output_96)))
        output_47 = torch.tanh(output_47)
        output_97 = inp[97]
        output_98 = inp[98]
        output_48 = self.encoder(torch.cat((output_97, output_98)))
        output_48 = torch.tanh(output_48)
        output_23 = self.encoder(torch.cat((output_47, output_48)))
        output_23 = torch.tanh(output_23)
        output_99 = inp[99]
        output_100 = inp[100]
        output_49 = self.encoder(torch.cat((output_99, output_100)))
        output_49 = torch.tanh(output_49)
        output_101 = inp[101]
        output_102 = inp[102]
        output_50 = self.encoder(torch.cat((output_101, output_102)))
        output_50 = torch.tanh(output_50)
        output_24 = self.encoder(torch.cat((output_49, output_50)))
        output_24 = torch.tanh(output_24)
        output_11 = self.encoder(torch.cat((output_23, output_24)))
        output_11 = torch.tanh(output_11)
        output_103 = inp[103]
        output_104 = inp[104]
        output_51 = self.encoder(torch.cat((output_103, output_104)))
        output_51 = torch.tanh(output_51)
        output_105 = inp[105]
        output_106 = inp[106]
        output_52 = self.encoder(torch.cat((output_105, output_106)))
        output_52 = torch.tanh(output_52)
        output_25 = self.encoder(torch.cat((output_51, output_52)))
        output_25 = torch.tanh(output_25)
        output_107 = inp[107]
        output_108 = inp[108]
        output_53 = self.encoder(torch.cat((output_107, output_108)))
        output_53 = torch.tanh(output_53)
        output_109 = inp[109]
        output_110 = inp[110]
        output_54 = self.encoder(torch.cat((output_109, output_110)))
        output_54 = torch.tanh(output_54)
        output_26 = self.encoder(torch.cat((output_53, output_54)))
        output_26 = torch.tanh(output_26)
        output_12 = self.encoder(torch.cat((output_25, output_26)))
        output_12 = torch.tanh(output_12)
        output_5 = self.encoder(torch.cat((output_11, output_12)))
        output_5 = torch.tanh(output_5)
        output_111 = inp[111]
        output_112 = inp[112]
        output_55 = self.encoder(torch.cat((output_111, output_112)))
        output_55 = torch.tanh(output_55)
        output_113 = inp[113]
        output_114 = inp[114]
        output_56 = self.encoder(torch.cat((output_113, output_114)))
        output_56 = torch.tanh(output_56)
        output_27 = self.encoder(torch.cat((output_55, output_56)))
        output_27 = torch.tanh(output_27)
        output_115 = inp[115]
        output_116 = inp[116]
        output_57 = self.encoder(torch.cat((output_115, output_116)))
        output_57 = torch.tanh(output_57)
        output_117 = inp[117]
        output_118 = inp[118]
        output_58 = self.encoder(torch.cat((output_117, output_118)))
        output_58 = torch.tanh(output_58)
        output_28 = self.encoder(torch.cat((output_57, output_58)))
        output_28 = torch.tanh(output_28)
        output_13 = self.encoder(torch.cat((output_27, output_28)))
        output_13 = torch.tanh(output_13)
        output_119 = inp[119]
        output_120 = inp[120]
        output_59 = self.encoder(torch.cat((output_119, output_120)))
        output_59 = torch.tanh(output_59)
        output_121 = inp[121]
        output_122 = inp[122]
        output_60 = self.encoder(torch.cat((output_121, output_122)))
        output_60 = torch.tanh(output_60)
        output_29 = self.encoder(torch.cat((output_59, output_60)))
        output_29 = torch.tanh(output_29)
        output_123 = inp[123]
        output_124 = inp[124]
        output_61 = self.encoder(torch.cat((output_123, output_124)))
        output_61 = torch.tanh(output_61)
        output_125 = inp[125]
        output_126 = inp[126]
        output_62 = self.encoder(torch.cat((output_125, output_126)))
        output_62 = torch.tanh(output_62)
        output_30 = self.encoder(torch.cat((output_61, output_62)))
        output_30 = torch.tanh(output_30)
        output_14 = self.encoder(torch.cat((output_29, output_30)))
        output_14 = torch.tanh(output_14)
        output_6 = self.encoder(torch.cat((output_13, output_14)))
        output_6 = torch.tanh(output_6)
        output_2 = self.encoder(torch.cat((output_5, output_6)))
        output_2 = torch.tanh(output_2)
        output_0 = self.encoder(torch.cat((output_1, output_2)))
        output_0 = torch.tanh(output_0)
        return output_0
'''

def gen_code(left, right, is_leaf, root):
    if is_leaf[root]:
        print(f"output_{root} = inp[{root}]")
    else:
        gen_code(left, right, is_leaf, left[root].item()) # (h,)
        gen_code(left, right, is_leaf, right[root].item()) # (h,)
        print(f"output_{root} = self.encoder(torch.cat((output_{left[root].item()}, output_{right[root].item()})))")
        print(f"output_{root} = torch.tanh(output_{root})")
        
if __name__ == '__main__':
    depth = 7
    n = 2 ** depth - 1
    # batch_id 25
    # parent [16, 16, 15, 15, 13, 11, 11, 10, 10, 14, 12, 12, 13, 14, 18, 17, 17, 18, -1]
    # root = 18
    # left = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 5, 10, 4, 9, 2, 0, 15, 14])
    # right = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 6, 11, 12, 13, 3, 1, 16, 17])
    # is_leaf = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # batch_id 36
    # parent [61, 60, 59, 59, 62, 55, 54, 54, 53, 53, 57, 51, 50, 49, 48, 47, 47, 46, 45, 45, 43, 42, 41, 40, 39, 38, 36, 35, 34, 34, 33, 33, 44, 37, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 64, 46, 52, 48, 49, 50, 51, 52, 58, 56, 55, 56, 57, 58, 63, 60, 61, 62, 63, 64, -1]
    root = 64
    left = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, 28, 27, 26, 33, 25, 24, 23, 22, 21, 20, 32, 18, 17, 15, 14, 13, 12, 11, 46, 8, 6, 5, 53, 10, 52, 2, 1, 0, 4, 58, 44])
    right = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 19, 45, 16, 47, 48, 49, 50, 51, 9, 7, 54, 55, 56, 57, 3, 59, 60, 61, 62, 63])
    is_leaf = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # left = torch.zeros(n, dtype=torch.int)
    # right = torch.zeros(n, dtype=torch.int)
    # is_leaf = torch.ones(n, dtype=torch.bool)
    # left[:n//2] = torch.arange(1, n, 2)
    # right[:n//2] = torch.arange(2, n, 2)
    # is_leaf[:n//2] = 0
    # root = 0

    gen_code(left, right, is_leaf, root)
