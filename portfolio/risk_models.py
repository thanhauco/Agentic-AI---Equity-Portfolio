"""
Advanced Risk Models for AlphaAgents.

Implements Hierarchical Risk Parity (HRP), Monte Carlo VaR, and tail-risk metrics.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Any, List, Optional
from loguru import logger

class RiskManager:
    """
    Advanced risk calculation engine.
    """
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95, method: str = 'monte_carlo', n_sims: int = 10000) -> float:
        """
        Calculate Value at Risk (VaR).
        """
        if method == 'historical':
            return -np.percentile(returns, (1 - confidence) * 100)
        
        elif method == 'monte_carlo':
            mu = returns.mean()
            sigma = returns.std()
            sim_returns = np.random.normal(mu, sigma, n_sims)
            return -np.percentile(sim_returns, (1 - confidence) * 100)
            
        return 0.0

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        """
        var = RiskManager.calculate_var(returns, confidence, method='historical')
        tail_returns = returns[returns <= -var]
        return -tail_returns.mean() if len(tail_returns) > 0 else var

    @staticmethod
    def get_hrp_weights(returns: pd.DataFrame) -> pd.Series:
        """
        Implement Hierarchical Risk Parity (HRP).
        """
        try:
            # 1. Compute correlation and distance matrix
            corr = returns.corr()
            dist = np.sqrt((1 - corr) / 2)
            
            # 2. Quasi-Diagonalization (Hierarchical Clustering)
            link = linkage(squareform(dist), method='single')
            
            # 3. Recursive Bisection
            def get_ivp(cov):
                ivp = 1. / np.diag(cov)
                ivp /= ivp.sum()
                return ivp

            def get_cluster_var(cov, c_items):
                cov_c = cov.iloc[c_items, c_items]
                w_ = get_ivp(cov_c)
                c_var = np.dot(np.dot(w_, cov_c), w_)
                return c_var

            def recursive_bisection(cov, sort_ix):
                w = pd.Series(1, index=sort_ix)
                c_list = [sort_ix]
                while len(c_list) > 0:
                    c_list = [c_list[i][j:k] for i in range(len(c_list)) 
                              for j, k in ((0, len(c_list[i]) // 2), 
                                          (len(c_list[i]) // 2, len(c_list[i])))
                              if len(c_list[i]) > 1]
                    for i in range(0, len(c_list), 2):
                        c0 = c_list[i]
                        c1 = c_list[i+1]
                        v0 = get_cluster_var(cov, c0)
                        v1 = get_cluster_var(cov, c1)
                        alpha = 1 - v0 / (v0 + v1)
                        w[c0] *= alpha
                        w[c1] *= 1 - alpha
                return w

            # Get sorted indices from link matrix
            def get_quasi_diag(link):
                link = link.astype(int)
                sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
                num_items = link[-1, 3]
                while sort_ix.max() >= num_items:
                    sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                    df0 = sort_ix[sort_ix >= num_items]
                    i = df0.index
                    j = df0.values - num_items
                    sort_ix[i] = link[j, 0]
                    df0 = pd.Series(link[j, 1], index=i + 1)
                    sort_ix = pd.concat([sort_ix, df0]).sort_index()
                    num_items = link.shape[0] + 1
                return sort_ix.tolist()

            sort_ix = get_quasi_diag(link)
            sort_ix = returns.columns[sort_ix].tolist()
            
            cov = returns.cov()
            weights = recursive_bisection(cov, sort_ix)
            return weights
            
        except Exception as e:
            logger.error(f"HRP Weight Calculation Error: {e}")
            return pd.Series(1/len(returns.columns), index=returns.columns)

