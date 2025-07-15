import os
import pdb

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

OPENML_DATACONFIG = {
    '/data/ytlee/Code2/new-tran/case_1/data_9m': {
        'bin': ["AlphaScore_Score_exist", "MetaSVM_Score_exist", "CADD_Score_exist", "Solv_acc_exist",
                "M-CAP_Score_exist", "DANN_Score_exist", "ClinPred_Score_exist", "MutationAssessor_Score_exist",
                "PolyPhen_Score_exist", "DEOGEN2_Score_exist", "fathmm_Score_exist", "MetaRNN_Score_exist",
                "MutationTaster_Score_exist", "MutPred_Score_exist", "BayesDel_addAF_Score_exist",
                "PrimateAI_Score_exist", "ESM1b_Score_exist", "BayesDel_noAF_Score_exist", "AlphaMissense_Score_exist",
                "Eigen_Score_exist", "gMVP_Score_exist", "MetaLR_Score_exist", "AF_exist"],
        'cat': ["AA_mut"],
        'num': ['AF', 'AF_exist', 'Res_wild_ALA', 'Res_wild_ARG', 'Res_wild_ASN', 'Res_wild_ASP',
                'Res_wild_CYS', 'Res_wild_GLN', 'Res_wild_GLU', 'Res_wild_GLY', 'Res_wild_HIS', 'Res_wild_ILE',
                'Res_wild_LEU', 'Res_wild_LYS', 'Res_wild_MET', 'Res_wild_PHE', 'Res_wild_PRO', 'Res_wild_SER',
                'Res_wild_THR', 'Res_wild_TRP', 'Res_wild_TYR', 'Res_wild_VAL', 'Res_mut_ALA', 'Res_mut_ARG',
                'Res_mut_ASN', 'Res_mut_ASP', 'Res_mut_CYS', 'Res_mut_GLN', 'Res_mut_GLU', 'Res_mut_GLY',
                'Res_mut_HIS', 'Res_mut_ILE', 'Res_mut_LEU', 'Res_mut_LYS', 'Res_mut_MET', 'Res_mut_PHE',
                'Res_mut_PRO', 'Res_mut_SER', 'Res_mut_THR', 'Res_mut_TRP', 'Res_mut_TYR', 'Res_mut_VAL',
                'B_factor', 'B_factor_exist', 'Res_contact_density_CACA_5A', 'HSE_u_proportion_5A',
                'HSE_d_proportion_5A', 'Res_contact_density_CACA_8A', 'HSE_u_proportion_8A', 'HSE_d_proportion_8A',
                'Res_contact_density_CACA_12A', 'HSE_u_proportion_12A', 'HSE_d_proportion_12A', 'Res_contact_density_CBCB_8A',
                'average_B_factor', 'Betweenness_centrality', 'Clustering_coefficient', 'Secondary_structure_is_H',
                'Secondary_structure_is_B', 'Secondary_structure_is_I', 'Secondary_structure_is_G', 'Secondary_structure_is_T',
                'Secondary_structure_is_S', 'Secondary_structure_is_E', 'Secondary_structure_is_C', 'Res_RASA',
                'Atom_type_is_C_6A_ball', 'Atom_type_is_CT_6A_ball', 'Atom_type_is_CA_6A_ball', 'Atom_type_is_N_6A_ball',
                'Atom_type_is_N2_6A_ball', 'Atom_type_is_N3_6A_ball', 'Atom_type_is_NA_6A_ball', 'Atom_type_is_O_6A_ball',
                'Atom_type_is_O2_6A_ball', 'Atom_type_is_OH_6A_ball', 'Atom_type_is_S_6A_ball', 'Atom_type_is_SH_6A_ball',
                'Partial_charge_6A_ball', 'Element_is_ANY_6A_ball', 'Element_is_C_6A_ball', 'Element_is_N_6A_ball',
                'Element_is_O_6A_ball', 'Element_is_S_6A_ball', 'Hydroxyl_6A_ball', 'Amide_6A_ball', 'Amine_6A_ball',
                'Carbonyl_6A_ball', 'Ring_system_6A_ball', 'Peptide_6A_ball', 'VDW_volume_6A_ball', 'Charge_6A_ball',
                'Neg_charge_6A_ball', 'Pos_charge_6A_ball', 'Charge_with_HIS_6A_ball', 'Hydrophobicity_6A_ball',
                'Mobility_6A_ball', 'Solvent_accessibility_6A_ball', 'Res_exist_ALA_6A_ball', 'Res_exist_ARG_6A_ball',
                'Res_exist_ASN_6A_ball', 'Res_exist_ASP_6A_ball', 'Res_exist_CYS_6A_ball', 'Res_exist_GLN_6A_ball',
                'Res_exist_GLU_6A_ball', 'Res_exist_GLY_6A_ball', 'Res_exist_HIS_6A_ball', 'Res_exist_ILE_6A_ball',
                'Res_exist_LEU_6A_ball', 'Res_exist_LYS_6A_ball', 'Res_exist_MET_6A_ball', 'Res_exist_PHE_6A_ball',
                'Res_exist_PRO_6A_ball', 'Res_exist_SER_6A_ball', 'Res_exist_THR_6A_ball', 'Res_exist_TRP_6A_ball',
                'Res_exist_TYR_6A_ball', 'Res_exist_VAL_6A_ball', 'Res_class1_is_HYDROPHOBIC_6A_ball', 'Res_class1_is_CHARGED_6A_ball',
                'Res_class1_is_POLAR_6A_ball', 'Res_class2_IS_NONPOLAR_6A_ball', 'Res_class2_IS_POLAR_6A_ball',
                'Res_class2_IS_BASIC_6A_ball', 'Res_class2_IS_ACIDIC_6A_ball', 'Secondary_structure1_is_3HELIX_6A_ball',
                'Secondary_structure1_is_4HELIX_6A_ball', 'Secondary_structure1_is_5HELIX_6A_ball', 'Secondary_structure1_is_BRIDGE_6A_ball',
                'Secondary_structure1_is_STRAND_6A_ball', 'Secondary_structure1_is_TURN_6A_ball', 'Secondary_structure1_is_BEND_6A_ball',
                'Secondary_structure1_is_COIL_6A_ball', 'Secondary_structure2_is_HELIX_6A_ball', 'Secondary_structure2_is_BETA_6A_ball',
                'Secondary_structure2_is_COIL_6A_ball', 'Atom_type_is_C_4.5A_ball', 'Atom_type_is_CT_4.5A_ball',
                'Atom_type_is_CA_4.5A_ball', 'Atom_type_is_N_4.5A_ball', 'Atom_type_is_N2_4.5A_ball', 'Atom_type_is_N3_4.5A_ball',
                'Atom_type_is_NA_4.5A_ball', 'Atom_type_is_O_4.5A_ball', 'Atom_type_is_O2_4.5A_ball', 'Atom_type_is_OH_4.5A_ball',
                'Atom_type_is_S_4.5A_ball', 'Atom_type_is_SH_4.5A_ball', 'Partial_charge_4.5A_ball', 'Element_is_ANY_4.5A_ball',
                'Element_is_C_4.5A_ball', 'Element_is_N_4.5A_ball', 'Element_is_O_4.5A_ball', 'Element_is_S_4.5A_ball',
                'Hydroxyl_4.5A_ball', 'Amide_4.5A_ball', 'Amine_4.5A_ball', 'Carbonyl_4.5A_ball', 'Ring_system_4.5A_ball',
                'Peptide_4.5A_ball', 'VDW_volume_4.5A_ball', 'Charge_4.5A_ball', 'Neg_charge_4.5A_ball', 'Pos_charge_4.5A_ball',
                'Charge_with_HIS_4.5A_ball', 'Hydrophobicity_4.5A_ball', 'Mobility_4.5A_ball', 'Solvent_accessibility_4.5A_ball',
                'Res_exist_ALA_4.5A_ball', 'Res_exist_ARG_4.5A_ball', 'Res_exist_ASN_4.5A_ball', 'Res_exist_ASP_4.5A_ball',
                'Res_exist_CYS_4.5A_ball', 'Res_exist_GLN_4.5A_ball', 'Res_exist_GLU_4.5A_ball', 'Res_exist_GLY_4.5A_ball',
                'Res_exist_HIS_4.5A_ball', 'Res_exist_ILE_4.5A_ball', 'Res_exist_LEU_4.5A_ball', 'Res_exist_LYS_4.5A_ball',
                'Res_exist_MET_4.5A_ball', 'Res_exist_PHE_4.5A_ball', 'Res_exist_PRO_4.5A_ball', 'Res_exist_SER_4.5A_ball',
                'Res_exist_THR_4.5A_ball', 'Res_exist_TRP_4.5A_ball', 'Res_exist_TYR_4.5A_ball', 'Res_exist_VAL_4.5A_ball',
                'Res_class1_is_HYDROPHOBIC_4.5A_ball', 'Res_class1_is_CHARGED_4.5A_ball', 'Res_class1_is_POLAR_4.5A_ball',
                'Res_class2_IS_NONPOLAR_4.5A_ball', 'Res_class2_IS_POLAR_4.5A_ball', 'Res_class2_IS_BASIC_4.5A_ball',
                'Res_class2_IS_ACIDIC_4.5A_ball', 'Secondary_structure1_is_3HELIX_4.5A_ball', 'Secondary_structure1_is_4HELIX_4.5A_ball',
                'Secondary_structure1_is_5HELIX_4.5A_ball', 'Secondary_structure1_is_BRIDGE_4.5A_ball',
                'Secondary_structure1_is_STRAND_4.5A_ball', 'Secondary_structure1_is_TURN_4.5A_ball',
                'Secondary_structure1_is_BEND_4.5A_ball', 'Secondary_structure1_is_COIL_4.5A_ball',
                'Secondary_structure2_is_HELIX_4.5A_ball', 'Secondary_structure2_is_BETA_4.5A_ball',
                'Secondary_structure2_is_COIL_4.5A_ball', 'Atom_type_is_C_4.5A_to_6A_shell', 'Atom_type_is_CT_4.5A_to_6A_shell',
                'Atom_type_is_CA_4.5A_to_6A_shell', 'Atom_type_is_N_4.5A_to_6A_shell', 'Atom_type_is_N2_4.5A_to_6A_shell',
                'Atom_type_is_N3_4.5A_to_6A_shell', 'Atom_type_is_NA_4.5A_to_6A_shell', 'Atom_type_is_O_4.5A_to_6A_shell',
                'Atom_type_is_O2_4.5A_to_6A_shell', 'Atom_type_is_OH_4.5A_to_6A_shell', 'Atom_type_is_S_4.5A_to_6A_shell',
                'Atom_type_is_SH_4.5A_to_6A_shell', 'Partial_charge_4.5A_to_6A_shell', 'Element_is_ANY_4.5A_to_6A_shell',
                'Element_is_C_4.5A_to_6A_shell', 'Element_is_N_4.5A_to_6A_shell', 'Element_is_O_4.5A_to_6A_shell',
                'Element_is_S_4.5A_to_6A_shell', 'Hydroxyl_4.5A_to_6A_shell', 'Amide_4.5A_to_6A_shell', 'Amine_4.5A_to_6A_shell',
                'Carbonyl_4.5A_to_6A_shell', 'Ring_system_4.5A_to_6A_shell', 'Peptide_4.5A_to_6A_shell', 'VDW_volume_4.5A_to_6A_shell',
                'Charge_4.5A_to_6A_shell', 'Neg_charge_4.5A_to_6A_shell', 'Pos_charge_4.5A_to_6A_shell', 'Charge_with_HIS_4.5A_to_6A_shell',
                'Hydrophobicity_4.5A_to_6A_shell', 'Mobility_4.5A_to_6A_shell', 'Solvent_accessibility_4.5A_to_6A_shell',
                'Res_exist_ALA_4.5A_to_6A_shell', 'Res_exist_ARG_4.5A_to_6A_shell', 'Res_exist_ASN_4.5A_to_6A_shell',
                'Res_exist_ASP_4.5A_to_6A_shell', 'Res_exist_CYS_4.5A_to_6A_shell', 'Res_exist_GLN_4.5A_to_6A_shell',
                'Res_exist_GLU_4.5A_to_6A_shell', 'Res_exist_GLY_4.5A_to_6A_shell', 'Res_exist_HIS_4.5A_to_6A_shell',
                'Res_exist_ILE_4.5A_to_6A_shell', 'Res_exist_LEU_4.5A_to_6A_shell', 'Res_exist_LYS_4.5A_to_6A_shell',
                'Res_exist_MET_4.5A_to_6A_shell', 'Res_exist_PHE_4.5A_to_6A_shell', 'Res_exist_PRO_4.5A_to_6A_shell',
                'Res_exist_SER_4.5A_to_6A_shell', 'Res_exist_THR_4.5A_to_6A_shell', 'Res_exist_TRP_4.5A_to_6A_shell',
                'Res_exist_TYR_4.5A_to_6A_shell', 'Res_exist_VAL_4.5A_to_6A_shell', 'Res_class1_is_HYDROPHOBIC_4.5A_to_6A_shell',
                'Res_class1_is_CHARGED_4.5A_to_6A_shell', 'Res_class1_is_POLAR_4.5A_to_6A_shell', 'Res_class2_IS_NONPOLAR_4.5A_to_6A_shell',
                'Res_class2_IS_POLAR_4.5A_to_6A_shell', 'Res_class2_IS_BASIC_4.5A_to_6A_shell', 'Res_class2_IS_ACIDIC_4.5A_to_6A_shell',
                'Secondary_structure1_is_3HELIX_4.5A_to_6A_shell', 'Secondary_structure1_is_4HELIX_4.5A_to_6A_shell',
                'Secondary_structure1_is_5HELIX_4.5A_to_6A_shell', 'Secondary_structure1_is_BRIDGE_4.5A_to_6A_shell',
                'Secondary_structure1_is_STRAND_4.5A_to_6A_shell', 'Secondary_structure1_is_TURN_4.5A_to_6A_shell',
                'Secondary_structure1_is_BEND_4.5A_to_6A_shell', 'Secondary_structure1_is_COIL_4.5A_to_6A_shell',
                'Secondary_structure2_is_HELIX_4.5A_to_6A_shell', 'Secondary_structure2_is_BETA_4.5A_to_6A_shell',
                'Secondary_structure2_is_COIL_4.5A_to_6A_shell', 'pLDDT', 'Atom_type_is_C_6A_ball_toAS', 'Atom_type_is_CT_6A_ball_toAS',
                'Atom_type_is_CA_6A_ball_toAS', 'Atom_type_is_N_6A_ball_toAS', 'Atom_type_is_N2_6A_ball_toAS', 'Atom_type_is_N3_6A_ball_toAS',
                'Atom_type_is_NA_6A_ball_toAS', 'Atom_type_is_O_6A_ball_toAS', 'Atom_type_is_O2_6A_ball_toAS', 'Atom_type_is_OH_6A_ball_toAS',
                'Atom_type_is_S_6A_ball_toAS', 'Atom_type_is_SH_6A_ball_toAS', 'Partial_charge_6A_ball_toAS', 'Element_is_ANY_6A_ball_toAS',
                'Element_is_C_6A_ball_toAS', 'Element_is_N_6A_ball_toAS', 'Element_is_O_6A_ball_toAS', 'Element_is_S_6A_ball_toAS',
                'Hydroxyl_6A_ball_toAS', 'Amide_6A_ball_toAS', 'Amine_6A_ball_toAS', 'Carbonyl_6A_ball_toAS', 'Ring_system_6A_ball_toAS',
                'Peptide_6A_ball_toAS', 'VDW_volume_6A_ball_toAS', 'Charge_6A_ball_toAS', 'Neg_charge_6A_ball_toAS', 'Pos_charge_6A_ball_toAS',
                'Charge_with_HIS_6A_ball_toAS', 'Hydrophobicity_6A_ball_toAS', 'Mobility_6A_ball_toAS', 'Solvent_accessibility_6A_ball_toAS',
                'Res_exist_ALA_6A_ball_toAS', 'Res_exist_ARG_6A_ball_toAS', 'Res_exist_ASN_6A_ball_toAS', 'Res_exist_ASP_6A_ball_toAS',
                'Res_exist_CYS_6A_ball_toAS', 'Res_exist_GLN_6A_ball_toAS', 'Res_exist_GLU_6A_ball_toAS', 'Res_exist_GLY_6A_ball_toAS',
                'Res_exist_HIS_6A_ball_toAS', 'Res_exist_ILE_6A_ball_toAS', 'Res_exist_LEU_6A_ball_toAS', 'Res_exist_LYS_6A_ball_toAS',
                'Res_exist_MET_6A_ball_toAS', 'Res_exist_PHE_6A_ball_toAS', 'Res_exist_PRO_6A_ball_toAS', 'Res_exist_SER_6A_ball_toAS',
                'Res_exist_THR_6A_ball_toAS', 'Res_exist_TRP_6A_ball_toAS', 'Res_exist_TYR_6A_ball_toAS', 'Res_exist_VAL_6A_ball_toAS',
                'Res_class1_is_HYDROPHOBIC_6A_ball_toAS', 'Res_class1_is_CHARGED_6A_ball_toAS', 'Res_class1_is_POLAR_6A_ball_toAS',
                'Res_class2_IS_NONPOLAR_6A_ball_toAS', 'Res_class2_IS_POLAR_6A_ball_toAS', 'Res_class2_IS_BASIC_6A_ball_toAS',
                'Res_class2_IS_ACIDIC_6A_ball_toAS', 'Secondary_structure1_is_3HELIX_6A_ball_toAS',
                'Secondary_structure1_is_4HELIX_6A_ball_toAS', 'Secondary_structure1_is_5HELIX_6A_ball_toAS',
                'Secondary_structure1_is_BRIDGE_6A_ball_toAS', 'Secondary_structure1_is_STRAND_6A_ball_toAS',
                'Secondary_structure1_is_TURN_6A_ball_toAS', 'Secondary_structure1_is_BEND_6A_ball_toAS',
                'Secondary_structure1_is_COIL_6A_ball_toAS', 'Secondary_structure2_is_HELIX_6A_ball_toAS',
                'Secondary_structure2_is_BETA_6A_ball_toAS', 'Secondary_structure2_is_COIL_6A_ball_toAS',
                'Atom_type_is_C_4.5A_ball_toAS', 'Atom_type_is_CT_4.5A_ball_toAS', 'Atom_type_is_CA_4.5A_ball_toAS',
                'Atom_type_is_N_4.5A_ball_toAS', 'Atom_type_is_N2_4.5A_ball_toAS', 'Atom_type_is_N3_4.5A_ball_toAS',
                'Atom_type_is_NA_4.5A_ball_toAS', 'Atom_type_is_O_4.5A_ball_toAS', 'Atom_type_is_O2_4.5A_ball_toAS',
                'Atom_type_is_OH_4.5A_ball_toAS', 'Atom_type_is_S_4.5A_ball_toAS', 'Atom_type_is_SH_4.5A_ball_toAS',
                'Partial_charge_4.5A_ball_toAS', 'Element_is_ANY_4.5A_ball_toAS', 'Element_is_C_4.5A_ball_toAS',
                'Element_is_N_4.5A_ball_toAS', 'Element_is_O_4.5A_ball_toAS', 'Element_is_S_4.5A_ball_toAS',
                'Hydroxyl_4.5A_ball_toAS', 'Amide_4.5A_ball_toAS', 'Amine_4.5A_ball_toAS', 'Carbonyl_4.5A_ball_toAS',
                'Ring_system_4.5A_ball_toAS', 'Peptide_4.5A_ball_toAS', 'VDW_volume_4.5A_ball_toAS', 'Charge_4.5A_ball_toAS',
                'Neg_charge_4.5A_ball_toAS', 'Pos_charge_4.5A_ball_toAS', 'Charge_with_HIS_4.5A_ball_toAS', 'Hydrophobicity_4.5A_ball_toAS',
                'Mobility_4.5A_ball_toAS', 'Solvent_accessibility_4.5A_ball_toAS', 'Res_exist_ALA_4.5A_ball_toAS',
                'Res_exist_ARG_4.5A_ball_toAS', 'Res_exist_ASN_4.5A_ball_toAS', 'Res_exist_ASP_4.5A_ball_toAS',
                'Res_exist_CYS_4.5A_ball_toAS', 'Res_exist_GLN_4.5A_ball_toAS', 'Res_exist_GLU_4.5A_ball_toAS',
                'Res_exist_GLY_4.5A_ball_toAS', 'Res_exist_HIS_4.5A_ball_toAS', 'Res_exist_ILE_4.5A_ball_toAS',
                'Res_exist_LEU_4.5A_ball_toAS', 'Res_exist_LYS_4.5A_ball_toAS', 'Res_exist_MET_4.5A_ball_toAS',
                'Res_exist_PHE_4.5A_ball_toAS', 'Res_exist_PRO_4.5A_ball_toAS', 'Res_exist_SER_4.5A_ball_toAS',
                'Res_exist_THR_4.5A_ball_toAS', 'Res_exist_TRP_4.5A_ball_toAS', 'Res_exist_TYR_4.5A_ball_toAS',
                'Res_exist_VAL_4.5A_ball_toAS', 'Res_class1_is_HYDROPHOBIC_4.5A_ball_toAS', 'Res_class1_is_CHARGED_4.5A_ball_toAS',
                'Res_class1_is_POLAR_4.5A_ball_toAS', 'Res_class2_IS_NONPOLAR_4.5A_ball_toAS', 'Res_class2_IS_POLAR_4.5A_ball_toAS',
                'Res_class2_IS_BASIC_4.5A_ball_toAS', 'Res_class2_IS_ACIDIC_4.5A_ball_toAS', 'Secondary_structure1_is_3HELIX_4.5A_ball_toAS',
                'Secondary_structure1_is_4HELIX_4.5A_ball_toAS', 'Secondary_structure1_is_5HELIX_4.5A_ball_toAS',
                'Secondary_structure1_is_BRIDGE_4.5A_ball_toAS', 'Secondary_structure1_is_STRAND_4.5A_ball_toAS',
                'Secondary_structure1_is_TURN_4.5A_ball_toAS', 'Secondary_structure1_is_BEND_4.5A_ball_toAS',
                'Secondary_structure1_is_COIL_4.5A_ball_toAS', 'Secondary_structure2_is_HELIX_4.5A_ball_toAS',
                'Secondary_structure2_is_BETA_4.5A_ball_toAS', 'Secondary_structure2_is_COIL_4.5A_ball_toAS',
                'Atom_type_is_C_4.5A_to_6A_shell_toAS', 'Atom_type_is_CT_4.5A_to_6A_shell_toAS', 'Atom_type_is_CA_4.5A_to_6A_shell_toAS',
                'Atom_type_is_N_4.5A_to_6A_shell_toAS', 'Atom_type_is_N2_4.5A_to_6A_shell_toAS', 'Atom_type_is_N3_4.5A_to_6A_shell_toAS',
                'Atom_type_is_NA_4.5A_to_6A_shell_toAS', 'Atom_type_is_O_4.5A_to_6A_shell_toAS', 'Atom_type_is_O2_4.5A_to_6A_shell_toAS',
                'Atom_type_is_OH_4.5A_to_6A_shell_toAS', 'Atom_type_is_S_4.5A_to_6A_shell_toAS', 'Atom_type_is_SH_4.5A_to_6A_shell_toAS',
                'Partial_charge_4.5A_to_6A_shell_toAS', 'Element_is_ANY_4.5A_to_6A_shell_toAS', 'Element_is_C_4.5A_to_6A_shell_toAS',
                'Element_is_N_4.5A_to_6A_shell_toAS', 'Element_is_O_4.5A_to_6A_shell_toAS', 'Element_is_S_4.5A_to_6A_shell_toAS',
                'Hydroxyl_4.5A_to_6A_shell_toAS', 'Amide_4.5A_to_6A_shell_toAS', 'Amine_4.5A_to_6A_shell_toAS',
                'Carbonyl_4.5A_to_6A_shell_toAS', 'Ring_system_4.5A_to_6A_shell_toAS', 'Peptide_4.5A_to_6A_shell_toAS',
                'VDW_volume_4.5A_to_6A_shell_toAS', 'Charge_4.5A_to_6A_shell_toAS', 'Neg_charge_4.5A_to_6A_shell_toAS',
                'Pos_charge_4.5A_to_6A_shell_toAS', 'Charge_with_HIS_4.5A_to_6A_shell_toAS', 'Hydrophobicity_4.5A_to_6A_shell_toAS',
                'Mobility_4.5A_to_6A_shell_toAS', 'Solvent_accessibility_4.5A_to_6A_shell_toAS', 'Res_exist_ALA_4.5A_to_6A_shell_toAS',
                'Res_exist_ARG_4.5A_to_6A_shell_toAS', 'Res_exist_ASN_4.5A_to_6A_shell_toAS', 'Res_exist_ASP_4.5A_to_6A_shell_toAS',
                'Res_exist_CYS_4.5A_to_6A_shell_toAS', 'Res_exist_GLN_4.5A_to_6A_shell_toAS', 'Res_exist_GLU_4.5A_to_6A_shell_toAS',
                'Res_exist_GLY_4.5A_to_6A_shell_toAS', 'Res_exist_HIS_4.5A_to_6A_shell_toAS', 'Res_exist_ILE_4.5A_to_6A_shell_toAS',
                'Res_exist_LEU_4.5A_to_6A_shell_toAS', 'Res_exist_LYS_4.5A_to_6A_shell_toAS', 'Res_exist_MET_4.5A_to_6A_shell_toAS',
                'Res_exist_PHE_4.5A_to_6A_shell_toAS', 'Res_exist_PRO_4.5A_to_6A_shell_toAS', 'Res_exist_SER_4.5A_to_6A_shell_toAS',
                'Res_exist_THR_4.5A_to_6A_shell_toAS', 'Res_exist_TRP_4.5A_to_6A_shell_toAS', 'Res_exist_TYR_4.5A_to_6A_shell_toAS',
                'Res_exist_VAL_4.5A_to_6A_shell_toAS', 'Res_class1_is_HYDROPHOBIC_4.5A_to_6A_shell_toAS',
                'Res_class1_is_CHARGED_4.5A_to_6A_shell_toAS', 'Res_class1_is_POLAR_4.5A_to_6A_shell_toAS',
                'Res_class2_IS_NONPOLAR_4.5A_to_6A_shell_toAS', 'Res_class2_IS_POLAR_4.5A_to_6A_shell_toAS',
                'Res_class2_IS_BASIC_4.5A_to_6A_shell_toAS', 'Res_class2_IS_ACIDIC_4.5A_to_6A_shell_toAS',
                'Secondary_structure1_is_3HELIX_4.5A_to_6A_shell_toAS', 'Secondary_structure1_is_4HELIX_4.5A_to_6A_shell_toAS',
                'Secondary_structure1_is_5HELIX_4.5A_to_6A_shell_toAS', 'Secondary_structure1_is_BRIDGE_4.5A_to_6A_shell_toAS',
                'Secondary_structure1_is_STRAND_4.5A_to_6A_shell_toAS', 'Secondary_structure1_is_TURN_4.5A_to_6A_shell_toAS',
                'Secondary_structure1_is_BEND_4.5A_to_6A_shell_toAS', 'Secondary_structure1_is_COIL_4.5A_to_6A_shell_toAS',
                'Secondary_structure2_is_HELIX_4.5A_to_6A_shell_toAS', 'Secondary_structure2_is_BETA_4.5A_to_6A_shell_toAS',
                'Secondary_structure2_is_COIL_4.5A_to_6A_shell_toAS'],

        'cols': ["AlphaScore_Score_exist", "MetaSVM_Score_exist", "CADD_Score_exist", "Solv_acc_exist",
                 "M-CAP_Score_exist", "DANN_Score_exist", "ClinPred_Score_exist", "MutationAssessor_Score_exist",
                 "PolyPhen_Score_exist", "DEOGEN2_Score_exist", "fathmm_Score_exist", "MetaRNN_Score_exist",
                 "MutationTaster_Score_exist", "MutPred_Score_exist", "BayesDel_addAF_Score_exist",
                 "PrimateAI_Score_exist", "ESM1b_Score_exist", "BayesDel_noAF_Score_exist", "AlphaMissense_Score_exist",
                 "Eigen_Score_exist", "gMVP_Score_exist", "MetaLR_Score_exist", "AF_exist",

                 "AlphaScore_Score", "MetaSVM_Score", "CADD_Score", "Solv_acc", "M-CAP_Score", "DANN_Score",
                 "ClinPred_Score", "MutationAssessor_Score", "PolyPhen_Score", "DEOGEN2_Score", "fathmm_Score",
                 "MetaRNN_Score", "MutationTaster_Score", "MutPred_Score", "BayesDel_addAF_Score", "PrimateAI_Score",
                 "ESM1b_Score", "BayesDel_noAF_Score", "AlphaMissense_Score", "Eigen_Score", "gMVP_Score",
                 "MetaLR_Score", "Residue_contact_density_CACA_5A", "hse_u_proportion_5A", "hse_d_proportion_5A",
                 "Residue_contact_density_CACA_8A", "hse_u_proportion_8A", "hse_d_proportion_8A",
                 "Residue_contact_density_CACA_12A", "hse_u_proportion_12A", "hse_d_proportion_12A",
                 "Residue_contact_density_CBCB_8A", "average_B_factor", "betweenness_centrality",
                 "clustering_coefficient", "Secondary_structure_is_H", "Secondary_structure_is_B",
                 "Secondary_structure_is_I", "Secondary_structure_is_G", "Secondary_structure_is_T",
                 "Secondary_structure_is_S", "Secondary_structure_is_E", "Secondary_structure_is_C",
                 "res_RASA", "pLDDT", "AF",
                 *["onehot1_" + aa for aa in "ARNDCQEGHILKMFPSTWYV"],
                 *["onehot2_" + aa for aa in "ARNDCQEGHILKMFPSTWYV"]],
        "binary_indicator": ["1"]
        }
}

def load_data(dataname, dataset_config=None, split=True, seed=123, nrows=None):
    if dataset_config is None:
        dataset_config = OPENML_DATACONFIG[dataname]

    if split == False:
        raise ValueError("Split option must be true to load the train and test datasets.")

    else:
        if os.path.exists(dataname):
            print(f'Load from local data dir {dataname}')
            # Load train and test datasets from specified CSV files
            filename_train = os.path.join(dataname, 'traindata1.csv')
            filename_test = os.path.join(dataname, 'testdata1.csv')

            # Load datasets
            train_df = pd.read_csv(filename_train, nrows=nrows)
            test_df = pd.read_csv(filename_test, nrows=nrows)

            # Extract labels and features
            y_train = train_df['Label']
            train_X = train_df.drop(['Label'], axis=1)  
            y_test = test_df['Label']
            test_X = test_df.drop(['Label'], axis=1)

            # 提取 AA_mut 列
            aa_mut_test = test_df['AA_mut']  # 提取 AA_mut 列

            num_cols = dataset_config['num']
            if len(num_cols) > 0:
                train_X[num_cols] = MinMaxScaler().fit_transform(train_X[num_cols])
                test_X[num_cols] = MinMaxScaler().fit_transform(test_X[num_cols])
            train_X = train_X[num_cols]
            test_X = test_X[num_cols]

            # Split the training dataset into training and validation sets
            train_dataset, val_dataset, y_train, y_val = train_test_split(
                train_X, y_train, test_size=0.2, random_state=seed, stratify=y_train, shuffle=True
            )

            feature_names = dataset_config['num']

            # 返回 AA_mut 列
            return (train_dataset, y_train), (val_dataset, y_val), (test_X, y_test, aa_mut_test), feature_names
