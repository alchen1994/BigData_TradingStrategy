import matplotlib.pyplot as plt
from os import walk

def exploreFolder(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for f in filenames:
            files.append(f.replace(".csv", ""))
        break
    return files

if __name__ == "__main__":
    
    files = exploreFolder('./semantic_data/add_fnc')
    # files.remove('.DS_Store')
   
    s_and_p = [x for x in files]
 
    s_and_p_test = sorted(s_and_p)

    reward_add_s = [0.013, -0.06, 0.292, -0.01, 0.224, 0.073, 0.12, 0.289, -0.113, 0.166, 0.078, 0.139, 0.267, 0.291, -0.054, 0.048, 0.029, 0.045, 0.009, 0.245, 0.289, 0.177, 0.281, 0.14, 0.093, -0.094, 0.055, 0.204, 0.053, 0.101, 0.066, 0.197, -0.059, -0.012, 0.93, 0.115, 0.2, 0.323, 0.01, 0.25, 0.097, 0.198]
    reward_add_fnc = [0.013, -0.052, 0.292, -0.01, 0.224, 0.073, 0.112, 0.289, -0.113, 0.166, 0.078, 0.139, 0.267, 0.291, -0.055, 0.048, 0.029, 0.045, 0.009, 0.245, 0.289, 0.177, 0.281, 0.14, 0.093, -0.094, 0.055, 0.204, 0.053, 0.078, 0.054, 0.192, -0.088, -0.012, 0.93, 0.155, 0.197, 0.323, 0.01, 0.25, 0.097, 0.198]
    reward_noadd = [0.021, -0.081, 0.308, -0.028, 0.208, 0.056, 0.102, 0.269, -0.149, 0.171, 0.079, 0.135, 0.273, 0.266, -0.062, 0.055, 0.036, 0.037, -0.001, 0.245, 0.315, 0.175, 0.276, 0.132, 0.064, -0.083, 0.051, 0.221, 0.042, 0.043, 0.059, 0.18, -0.124, -0.003, 0.959, 0.141, 0.195, 0.315, -0.016, 0.241, 0.095, 0.188]


    plt.plot(s_and_p_test[:20],reward_add_fnc[:20],color = 'tab:purple')
    # plt.plot(s_and_p_test[:20],reward_add_s[:20],color = 'tab:red')
    plt.plot(s_and_p_test[:20],reward_noadd[:20],color = 'tab:gray')

    plt.title("DQN_add_VIX vs DQN_noadd")
    # plt.title("DQN_add_Semantic vs DQN_noadd")
    plt.xlabel("stock")
    plt.ylabel("return")
    plt.legend(['DQN_add_VIX', 'DQN_noadd'], loc='upper left')
    # plt.legend(['DQN_add_Semantic', 'DQN_noadd'], loc='upper left')
    plt.show()

  











