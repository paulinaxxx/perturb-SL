import pandas as pd
import numpy as np
import scanpy as sc


def calc_viability(exprs_df):
    achilles_param = pd.read_csv("data/achilles_model.csv")

    intersect_gene = set(list(exprs_df.columns)).intersection(set(list(achilles_param["gene"])))
    
    df_achilles = exprs_df[list(intersect_gene)]
    achilles_param = achilles_param.set_index("gene")
    achilles_reindex = achilles_param.loc[list(intersect_gene)]

    predict_viability = np.dot(df_achilles.to_numpy(), achilles_reindex.to_numpy())

    return predict_viability


if __name__ == "__main__":
    
    # norman19
    # filepath = 'data/h5ad/NormanWeissman2019_filtered.h5ad'
    # adata = sc.read_h5ad(filepath)
    # sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)

    # df = adata.to_df()
    # norm_viab = calc_viability(df)
    
    # reploge22
    filepath = 'data/h5ad/ReplogleWeissman2022_K562_essential.h5ad'
    adata = sc.read_h5ad(filepath)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    df = adata.to_df()
    replogle_viab = calc_viability(df)
    print(replogle_viab[:10])

    np.savetxt("./result/replogle22_predict_viability.txt", replogle_viab)
    adata.obs["viability"] = replogle_viab

    # gasperini19

    

    