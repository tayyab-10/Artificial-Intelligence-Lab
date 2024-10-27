# def balance_brackets(expression):
#     open_needed = 0  
#     close_needed = 0  
    
#     for char in expression:
#         if char == '(':
#             close_needed += 1 
#         elif char == ')':
#             if close_needed > 0:
#                 close_needed -= 1  
#             else:
#                 open_needed += 1 
    
    
#     balanced_expression = '(' * open_needed + expression + ')' * close_needed
#     return balanced_expression

# # Test 
# expression = "(a+b(c)"
# balanced_expression = balance_brackets(expression)
# print(f"Input: {expression}")
# print(f"Output: {balanced_expression}")

def balance_brackets_dp(expression):
    n = len(expression)
    
    # dp[i][j] will store the minimum number of insertions required to balance expression[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Fill the DP table by considering substrings of increasing lengths
    for length in range(2, n + 1):  # length of the substring
        for i in range(n - length + 1):
            j = i + length - 1
            
            if expression[i] == '(' and expression[j] == ')':
                dp[i][j] = dp[i + 1][j - 1]  # No extra insertions needed, this pair is balanced
            else:
                # Try adding '(' to balance i or ')' to balance j, take the minimum of the two
                dp[i][j] = min(dp[i + 1][j] + (expression[i] != '('),   # Balance i
                               dp[i][j - 1] + (expression[j] != ')'))   # Balance j
    
    # Now use dp[0][n-1] to construct the balanced expression
    result = []
    i, j = 0, n - 1
    
    while i <= j:
        if expression[i] == '(' and expression[j] == ')':
            # This is already a balanced pair, so include both
            result.append(expression[i])
            i += 1
            if i <= j:
                result.append(expression[j])
                j -= 1
        elif dp[i + 1][j] + (expression[i] != '(') <= dp[i][j - 1] + (expression[j] != ')'):
            # Add '(' if expression[i] is not already '('
            if expression[i] != '(':
                result.append('(')
            result.append(expression[i])
            i += 1
        else:
            # Add ')' if expression[j] is not already ')'
            if expression[j] != ')':
                result.append(')')
            result.append(expression[j])
            j -= 1

    # Convert result list to a string
    return ''.join(result)


# Test the function
expression = "(a+b(c)"
balanced_expression = balance_brackets_dp(expression)
print(f"Input: {expression}")
print(f"Output: {balanced_expression}")
