import math

def pooled_stats(m1, s1, n1, m2, s2, n2):
    # m1, s1, n1: mean, sample sd, size of group 1
    # m2, s2, n2: mean, sample sd, size of group 2
    # return (combined_mean, combined_sd)
    N = n1 + n2
    combined_mean = (n1 * m1 + n2 * m2) / N

    # Total sum of squares = within-group SS + between-group SS
    ss_within = (n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)
    ss_between = n1 * ((m1 - combined_mean) ** 2) + n2 * ((m2 - combined_mean) ** 2)

    combined_var = (ss_within + ss_between) / (N - 1)
    combined_sd = math.sqrt(combined_var)

    return (float(combined_mean), float(combined_sd))