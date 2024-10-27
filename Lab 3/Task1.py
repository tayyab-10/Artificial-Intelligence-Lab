# import queue

# q = queue.Queue()
# count = 0

# def EditDistance(str1, str2):
#     global count 
    
#     for i in range(len(str1)):
#         if str1[i] == str2[i]:
#             q.put(str1[i])  
#         else:
#             q.put(str2[i])  
#             count += 1  

   
#     for i in range(len(str1), len(str2)):  // iterate through extra characters
#         q.put(str2[i]) 
#         count += 1  
#     return count


# # Print the queue contents
# result = ""
# while not q.empty():
#     result += q.get()

# print("Queue contents (edited string):", result)


def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    
 
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
   
    for i in range(m+1):
        for j in range(n+1):
            
            if i == 0:    #kitten and Sitting 
                dp[i][j] = j    
            elif j == 0:
                dp[i][j] = i    
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
          
            else:
                dp[i][j] = 1 + min(dp[i][j-1],    # Insert
                                   dp[i-1][j],    # Remove
                                   dp[i-1][j-1])  # Replace
    
    return dp[m][n]

str1 = input("Enter First String: ")
str2 = input("Enter Second String: ")

edit_distance = edit_distance(str1, str2)
print(f"Edit distance between '{str1}' and '{str2}' is:", edit_distance)