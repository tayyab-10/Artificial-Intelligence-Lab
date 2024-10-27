def min_coins(coins, target):
    
    dp = [float('inf')] * (target + 1)
    dp[0] = 0 

    # Fill the dp table
    for coin in coins:
        for i in range(coin, target + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[target] if dp[target] != float('inf') else -1

# Example usage
coins = [1, 2, 5] 
target = 11        
result = min_coins(coins, target)

if result != -1:
    print(f"The minimum number of coins required to make {target} is: {result}")
else:
    print(f"It's not possible to make {target} with the given denominations.")
