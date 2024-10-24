
  		  	   		  		 		  		  		    	 		 		   		 		  
import math  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		  		 		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		  		 		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  		 		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		  		 		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    #This should generate a multiple linear regression. So LinRegLearner should outperform a DTLearner
    np.random.seed(seed)
    col = np.random.choice(range(2, 11))
    row = np.random.choice(range(10, 21))
    X = np.random.rand(row, col) # give every value in X a random number
    Y = np.mean(X, axis=1)  # mean of each row, this is linear

    return X, Y
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  		 		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		  		 		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    #This should generate a non linear regression. So DTLearner should outperform a LinRegLearner
    np.random.seed(seed)
    col = np.random.choice(range(2, 11))
    row = np.random.choice(range(10, 21))
    X = np.random.rand(row, col)  # give every value in X a random number
    Y = np.ones(row) #default value is 1
    for i in range(row):
        if X[i, 0] * row > X[:, 1].sum():
            Y[i] = X[i, -1] ** 2 - 1 # x[i,-1]^(n) +e, make it non-linear

    return X, Y
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def author():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    return "ytsang6"  # Change this to your user ID
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("they call me Tim.")
    #x, y = best_4_lin_reg()
    #x, y = best_4_dt()

    #print(np.random.choice(range(2, 11)))
    #print(np.random.choice(range(10, 101)))
