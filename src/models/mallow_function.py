import math
import numpy as np
import sys as sys
import random 
from typing import List, Optional, Dict, Any
from collections import defaultdict

# Add the path to the utility modules
# Adjust this path based on your directory structure
sys.path.append('../src/utils')  # Ensure this path points to the directory containing your utility modules

from po_fun import BasicUtils, StatisticalUtils


class Mallows:
    @staticmethod
    def mallows_local_factor(i, theta, y):

        if len(y) <= 1:
            return 1.0  # If there's only one item, the local factor is 1

        # Count how many items in `y` are positioned before `i` in `y` 
        # even though `i` has a smaller index in `y`.
        idx_i = y.index(i)
        d = 0
        for a in y:
            if a != i:
                if idx_i > y.index(a):
                    d += 1

        # The denominator is sum_{k=0..(|y|-1)} e^{-theta * k}, for length(y) items
        n = len(y)
        denom = 0.0
        for k in range(n):
            denom += math.exp(-theta * k)

        numerator = math.exp(-theta * d)
        return numerator / denom
    @staticmethod
    def p_mallows_of_l_given_y(l, y, theta):
        """
        A basic function for p^{(M)}(l | y, theta) if we define it as a product of local factors
        or use the exponential form. We'll do local factor approach:
        p^{(M)}(l|y,theta) = ∏_{i=1..n} q^{(M)}(l_i | y_{i..n}, theta).
        We'll do a sequential interpretation: 
        we build l in order, at step i pick l[i], multiply local factor.
        """
        if len(l) != len(y):
            # if we interpret partial, or mismatch => return 0
            return 0.0

        prob_val = 1.0
        remain = list(y)  # items that haven't been "used" yet
        for item in l:
            if item not in remain:
                return 0.0
            # local factor
            gf = Mallows.mallows_local_factor(item, theta,remain)
            prob_val *= gf
            # remove 'item'
            remain.remove(item)

        return prob_val
    
    @staticmethod
    def f_mallows(y, h, theta, O_i_indice):
        """
        We pass 'O_i_indice' as the list of item labels corresponding
        to the rows/columns in 'h'. So h[i,:] corresponds to O_i_indice[i].
        """
        n = h.shape[0]
        # 1) If total => exactly 1 extension => p(l|y,theta)
        if BasicUtils.is_total_order(h):
            # single extension
            l = GenerationUtils.topological_sort(h)
            # note that 'l' is in 0..n-1, but we interpret them as O_i_indice[l[i]]
            # let's convert that to actual labels
            real_l = [O_i_indice[x] for x in l]
            p_val = Mallows.p_mallows_of_l_given_y(real_l, y, theta)
            return p_val

        # 2) If h is empty => sum=1
        if np.sum(h) == 0:
            return 1.0

        # 3) Sum over 'tops'
        tops = BasicUtils.find_tops(h)
        f_val = 0.0
        for t in tops:
            k_label = O_i_indice[t]    # the actual item label
            # local factor
            gk = Mallows.mallows_local_factor(k_label, theta, y)

            # remove row t, col t
            h_sub = np.delete(np.delete(h, t, axis=0), t, axis=1)
            # remove item from O_i_indice
            O_i_sub = O_i_indice[:t] + O_i_indice[t+1:]

            # remove k_label from y
            y_sub = [itm for itm in y if itm != k_label]

            f_k = Mallows.f_mallows(y_sub, h_sub, theta, O_i_sub)
            f_val += gk * f_k

        return f_val


    @staticmethod
    def compute_mallows_likelihood(y, h, theta, O_i_indice):
        """
        p^{(M)}(y | h, theta) = f / count
        """
        f_val = Mallows.f_mallows(y, h, theta, O_i_indice)
        tr_h = BasicUtils.transitive_reduction(h)
        c_val = BasicUtils.nle(tr_h)
        if c_val == 0:
            return 0.0
        return f_val / c_val
    
    @staticmethod
    def generate_total_order_noise_mallow(y: List[int],
                                            h: np.ndarray,
                                            theta: float,
                                            O_indices: List[int]) -> List[int]:
        """
        Recursively generate a total order from the given choice set and partial order h
        using the pure Mallows model (i.e., without any jump probability).

        Parameters:
        y: List of item labels (or identifiers) that remain to be ordered.
        h: The partial order (adjacency matrix) corresponding to the items in O_indices.
        theta: The Mallows parameter.
        O_indices: The list of item labels corresponding to the rows/columns of h.
        
        Returns:
        A total order (list of item labels) generated under the Mallows model.
        """
        # Base cases:
        if len(y) == 0:
            return []
        if len(y) == 1:
            return y

        # Determine indices in the current ordering that are "tops" of the partial order.
        # If the function BasicUtils.find_tops is defined, use it; otherwise, use all indices.
        if hasattr(BasicUtils, 'find_tops'):
            tops = BasicUtils.find_tops(h)
        else:
            tops = list(range(len(y)))

        # For each candidate (by its local index in the remaining set), compute the local factor.
        candidate_probs = []
        for local_idx in tops:
            candidate = O_indices[local_idx]
            gf = Mallows.mallows_local_factor(candidate, theta, y)
            candidate_probs.append(gf)
        
        # Normalize the probabilities.
        total_prob = sum(candidate_probs)
        candidate_probs = [p / total_prob for p in candidate_probs]

        # Sample one candidate from the top candidates using these probabilities.
        chosen_top_idx = random.choices(tops, weights=candidate_probs, k=1)[0]
        chosen_item = O_indices[chosen_top_idx]

        # Remove the chosen candidate from the current ordering.
        y_new = [item for item in y if item != chosen_item]
        # Remove the corresponding row and column from h.
        h_new = np.delete(np.delete(h, chosen_top_idx, axis=0), chosen_top_idx, axis=1)
        # Remove the chosen candidate from O_indices.
        O_new = O_indices[:chosen_top_idx] + O_indices[chosen_top_idx+1:]

        # Recursively generate the order for the remaining items.
        return [chosen_item] + StatisticalUtils.generate_total_order_mallows_no_jump(y_new, h_new, theta, O_new)