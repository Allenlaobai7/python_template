def longest_common_substring(s1, s2):
    longest = ''
    for i in range(len(s1)):
        tmp = s2
        while len(tmp)>len(longest) and len(s1[i:])>len(longest):
            i2 = i  # will be updated
            longest_tmp = ''
            idx = tmp.find(s1[i])
            if idx != -1:
                tmp = tmp[idx:]
                for j in range(min(len(tmp), len(s1[i:]))):
                    # print(tmp, j, i2, s1[i2], tmp[j], longest_tmp, longest)
                    if tmp[j] == s1[i2]:
                        longest_tmp = longest_tmp + tmp[j]
                        if len(longest_tmp) > len(longest):
                            longest = longest_tmp
                        i2 += 1
                    else:
                        break
                tmp = tmp[1:]  # rm the current matching prefix
            else:
                break
        i += 1
    return longest

# print(longest_common_substring('abcdeabcabcedefabd', 'abc bcd cbd abc abcadc efdab'))


def longestPalindrome(s):
    def get_max_len(s, left, right):
        length = len(s)
        while left >= 0 and right < length and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    if len(s) <= 1:
        return s
    start = end = 0
    length = len(s)
    for i in range(length):
        max_len_1 = get_max_len(s, i, i + 1)
        max_len_2 = get_max_len(s, i, i)
        max_len = max(max_len_1, max_len_2)
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    return s[start: end + 1]
# print(longestPalindrome("aacabdkacaa"))

def largest_area(l):
    ret = 0
    left, right = 0, len(l)-1
    while left < right:
        area = (right - left) * min(l[left], l[right])
        if area > ret:
            ret = area
        if l[left] < l[right]:
            left += 1
        else:
            right -= 1
    return ret
# print(largest_area([1,8,6,2,5,4,8,3,7]))

def twoSum(l, target):
    sols = []
    potential = []
    for i, v in enumerate(l):
        remaining = target - v
        if remaining in potential:
            sols.append([v, remaining])
        else:
            potential.append(v)
    return sols

def twoSum(l, target):
    sols = []
    l = sorted(l)
    left, right = 0, len(l)-1
    while left < right:
        print(left, right)
        total = l[left] + l[right]
        if total == target:
            sols.append([l[left], l[right]])
            left += 1
            right -= 1
            while left < right and (l[left] == l[left-1]):  # skip same num
                left += 1
            while left < right and (l[right] == l[right + 1]):  # skip same num
                right -= 1
        elif total < target:
            left += 1
            while left < right and (l[left] == l[left - 1]):  # skip same num
                left += 1
        else:  # total > target
            right -= 1
            while left < right and (l[right] == l[right + 1]):  # skip same num
                right -= 1
    return sols
# print(twoSum([-1,0,1,2,-1,-4,-2,-3,3,0,4], 2))

def threeSum(nums):
    ret = []
    if len(nums) >= 3:
        nums = sorted(nums)
        i = 0
        while i <(len(nums) - 2):
            j, k = i + 1, len(nums) - 1
            target = 0 - nums[i]
            while j < k:
                total = nums[j] + nums[k]
                if total == target:
                    ret.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k and (nums[j] == nums[j - 1]):  # skip same j
                        j += 1
                    while j < k and (nums[k] == nums[k + 1]):  # skip same k
                        k -= 1
                elif total < target:
                    j += 1
                    while j < k and (nums[j] == nums[j - 1]):  # skip same j
                        j += 1
                else:  # total > target
                    k -= 1
                    while j < k and (nums[k] == nums[k + 1]):  # skip same k
                        k -= 1
            i += 1
            while i < (len(nums) -1) and (nums[i] == nums[i - 1]):  # skip same i
                i += 1
    return ret
# print(threeSum([-1,0,1,2,-1,-4,-2,-3,3,0,4]))

def threeSumClosest(nums, target):
    ret = sum(nums[:3])
    diff = abs(target - ret)
    if len(nums) >= 3:
        nums = sorted(nums)
        i = 0
        while i <(len(nums) - 2):
            j, k = i + 1, len(nums) - 1
            while j < k:
                threesum = nums[i] + nums[j] + nums[k]
                threesumdiff = target - threesum
                if abs(threesumdiff) < diff:
                    ret = threesum
                    diff = abs(threesumdiff)
                if threesumdiff > 0:
                    j += 1
                    while j < k and (nums[j] == nums[j - 1]):  # skip same j
                        j += 1
                else:  # total > target
                    k -= 1
                    while j < k and (nums[k] == nums[k + 1]):  # skip same k
                        k -= 1
            i += 1
            while i < (len(nums) -1) and (nums[i] == nums[i - 1]):  # skip same i
                i += 1
    return ret
# print(threeSumClosest([0,2,1,-3], 1))


def fourSum(nums, target):
    def get_sum(start_idx, target, n):
        ret = []
        if n > 2:
            while start_idx <= (len(nums) - n):
                res = get_sum(start_idx + 1, target - nums[start_idx], n - 1)
                if res:
                    for sol in res:
                        sol.append(nums[start_idx])  # add current value
                        ret.append(sol)
                start_idx += 1
                while start_idx <= (len(nums) - n) and (nums[start_idx] == nums[start_idx - 1]):  # skip same num
                    start_idx += 1
        else:
            j, k = start_idx, len(nums) - 1
            while j < k:
                total = nums[j] + nums[k]
                if total == target:
                    ret.append([nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k and (nums[j] == nums[j - 1]):  # skip same num
                        j += 1
                    while j < k and (nums[k] == nums[k + 1]):  # skip same num
                        k -= 1
                elif total < target:
                    j += 1
                    while j < k and (nums[j] == nums[j - 1]):  # skip same num
                        j += 1
                else:  # total > target
                    k -= 1
                    while j < k and (nums[k] == nums[k + 1]):  # skip same num
                        k -= 1
        return ret

    ret = []
    if len(nums) >= 4:
        nums = sorted(nums)
        start_idx = 0
        ret = get_sum(start_idx, target, 4)
    return ret
# print(fourSum([1,0,-1,0,-2,2], 0))

def generateParenthesis(n):
    def generate(out, s, open, close):
        if open == close == 0:
            out.append(s)
        elif open == 0:  # must close
            generate(out, s + ')', open, close - 1)
        elif open == close:  # must open
            generate(out, s + '(', open - 1, close)
        else:  # can open or close
            generate(out, s + ')', open, close - 1)
            generate(out, s + '(', open - 1, close)
        return out
    ret = generate([], '', n, n)
    return ret
# print(generateParenthesis(3))

def strStr(haystack, needle):
    if needle == '':
        return 0
    else:
        res = -1
        needle_length = len(needle)
        for i in range(0, len(haystack) - len(needle) + 1):
            if haystack[i: i + needle_length] == needle:
                res = i
                break
        return res
# print(strStr("aabbbad","bad"))

def isValidSudoku(board):
    def validLst(lst):
        print(lst)
        isValid = True
        seen = set()
        for item in lst:
            if item != '.':
                if item in seen:
                    isValid = False
                    break
                else:
                    seen.add(item)
        return isValid
    ret = True
    if any(validLst(board[i]) == False for i in range(0, len(board))):
        ret = False
    elif any(validLst(list(zip(*board))[i]) == False for i in range(0, len(board))):
        ret = False
    else:
        for i in (0, 3, 6):
            for j in (0, 3, 6):
                square = [x[j:j+3] for x in board[i:i+3]]
                square = [value for v in square for value in v]  # flatten lst
                if validLst(square) == False:
                    ret = False
                    break
    return ret

# def firstMissingPositive(nums):
#     lst = list(range(1, len(nums)+1))
#     for i in nums:
#         lst
#     print(lst)

def firstMissingPositive(nums):
    nums = sorted(nums)
    i, integer = 0, 1
    while i < len(nums):
        if (nums[i] <= 0):
            i += 1
        else:
            if nums[i] != integer:
                break
            else:
                integer += 1
                i += 1
                while (i < len(nums)) and (nums[i] == nums[i - 1]):  # duplicates
                    i += 1
    return integer
# print(firstMissingPositive([1,2,0,1]))

def trap(height):
    out, left_max, left_max_array, right_max, right_max_array = 0, 0, [], 0, []
    for h in height:
        left_max = max(left_max, h)
        left_max_array.append(left_max)
    for h in height[::-1]:
        right_max = max(right_max, h)
        right_max_array.append(right_max)
    right_max_array = right_max_array[::-1]  # inverted
    for i, v in enumerate(height):
        left_max, right_max = left_max_array[i], right_max_array[i]
        if v < left_max and v <  right_max:
            out += min(left_max, right_max) - v
    return out
# print(trap([4,2,0,3,2,5]))

# def is_match(s, p):
#     i = 0
#     while (i < len(p)) and (i < len(s)):
#         print(i, s, p)
#         if p[i] == '*':
#             if len(p) == 1:  # only * remaining
#                 return True
#             else:
#                 j = 0
#                 while i + j < len(s):
#                     if is_match(s[i+j:], p[i+1:]):
#                         return True
#                     j += 1
#                 return False
#         elif (p[i] == '?') or (s[i] == p[i]):
#             return is_match(s[i + j:], p[i + 1:])
#         else:
#             return False
#     if (len(s) > len(p)) and p[-1] != '*':
#         if len(s) == 1:
#
#         return False
#     if len(s) == 0:
#         if len(p) == 1:
#             if p[i] == '*':
#                 return True
#             else:
#                 return False
#         else:
#             return is_match(s, p[i + 1:])
# print(is_match('adceb', '*a*bb'))

def reverseString(s):
    # for i in range(0, len(s)//2):
    #     reversed_idx = (i+1) * (-1)
    #     tmp = s[reversed_idx]
    #     s[reversed_idx] = s[i]
    #     s[i] = tmp
    # return s
    # return s[::-1]
    out = []
    for i in range(len(s)-1, 0-1, -1):
        out.append(s[i])
    return out
# print(reverseString(["h","e","l","l","o"]))

def singleNumber(nums):
    if len(nums) == 1:
        return nums[0]
    else:
        nums = sorted(nums)
        i = 1
        while i < len(nums):
            if nums[i] == nums[i-1]:
                i += 2
                if i >= len(nums):
                    return nums[-1]  # the last one is single
            else:
                return nums[i-1]
# print(singleNumber([4,1,2,1,2]))

def fizzBuzz(n):
    return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n+1)]
# print(fizzBuzz(3))

# def majorityElement(nums):
#     if len(nums) == 1:
#         return nums[0]
#     else:
#         counter = {}
#         for i in nums:
#             if i in counter:
#                 counter[i] += 1
#             else:
#                 counter[i] = 1
#         for k, v in counter.items():
#             if v >= (len(nums) + 1) // 2:
#                 return k

def majorityElement(nums):
    if len(nums) == 1:
        return nums[0]
    else:
        nums = sorted(nums)
        length = (len(nums) + 1) // 2
        cnt, i = 1, 0
        while i < len(nums)-1:
            while (nums[i] == nums[i+1]):
                i += 1
                cnt += 1
                if cnt >= length:
                    return nums[i]
            cnt = 1
            i += 1
# print(majorityElement([3,2,3]))

def isAnagram(s, t):
    counter = {}
    for item in s:
        counter[item] = counter.get(item, 0) + 1
    for item in t:
        counter[item] = counter.get(item, 0) - 1
    if all(i == 0 for i in counter.values()):
        return True
    else:
        return False
# print(isAnagram("anagram", "nagaram"))

def moveZeroes(nums):
    first_zero = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[i], nums[first_zero] = nums[first_zero], nums[i]
            first_zero += 1
    return nums
# print(moveZeroes([2,0,0,1,0,0,1]))

def titleToNumber(columnTitle):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = range(1, 27)
    letter_to_number = {v[0]: v[1] for v in list(zip(letters, numbers))}
    # for i in range(0, title_length):
    #     res += letter_to_number[columnTitle[i]] * 26 ** (title_length - i - 1)
    # return res
    res = letter_to_number[columnTitle[0]]
    if len(columnTitle)>1:
        for i in columnTitle[1:]:
            res = 26 * res + letter_to_number[i]
    return res
# print(titleToNumber('AB'))

def containsDuplicate(nums):
    # return len(set(nums)) != len(nums)
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]:
            return True
    return False

def generate(numRows):  # pascal's triangle
    if numRows == 1:
        return [[1]]
    elif numRows == 2:
        return [[1], [1,1]]
    else:
        i = 3
        res = [[1], [1,1]]
        while i <= numRows:
            newline = list(map(lambda x, y: x + y, res[-1][:-1], res[-1][1:]))
            res.append([1] + newline + [1])
            i += 1
        return res
# print(generate(5))

def missingNumber(nums):
    return sum(range(0, len(nums) + 1)) - sum(nums)

def permute(nums):
    res, tmp = [[nums[0]]], []
    for i, num in enumerate(nums[1:]):
        tmp = res
        res = []
        for item in tmp:
            for j in range(i+2):
                print(j)
                res.append(item[:j] + [num] + item[j:])
    return res
# print(permute([1,2,3]))
# can use reduce, recursive

def firstUniqChar(s):
    counter, repeated = {}, []
    for i, v in enumerate(s):
        if v not in repeated:
            if v not in counter:
                counter[v] = i
            else:
                repeated.append(v)
                del counter[v]
    return min(counter.values()) if counter else -1
# print(firstUniqChar("loveleetcode"))

def intersect(nums1, nums2):
    res = []
    for i in nums1:
        if i in nums2:
            res.append(i)
            nums2.remove(i)
    return res
# print(intersect([1,2,2,1], [2,2]))
# can use sort first and then two pointer; or collection counter two nums and then add to res

def isHappy(n):
    def end(n, loop_lst):
        if n == 1:
            return True
        else:
            n = sum([int(i) ** 2 for i in str(n)])
            if n in loop_lst:
                return False
            else:
                loop_lst.append(n)
                return end(n, loop_lst)
    return end(n, [n])
# print(isHappy(7))

def searchRange(nums, target):
    start, end = -1, -1
    for i in range(len(nums)):
        if nums[i] == target:
            start = i
            while (i < len(nums) and nums[i] == target):
                i += 1
            end = i - 1
            break
    return [start, end]
# print(searchRange([5,7,7,8,8,10], 8))

def longestValidParentheses(s):  # using stack
    stack, result = [(-1, ')')], 0
    for i, paren in enumerate(s):
        if paren == ')' and stack[-1][1] == '(':
            stack.pop()
            result = max(result, i - stack[-1][0])
        else:
            stack += (i, paren),
    return result
# print(longestValidParentheses('()(()'))

def subsets(nums):
    res = [[]]
    for num in nums:
        res += [item + [num] for item in res]
    return res
# print(subsets([1,2,3]))

def topKFrequent(nums, k):
    # from collections import Counter
    # return [i[0] for i in Counter(nums).most_common(k)]

    # counter = {}
    # for i in nums:
    #     counter[i] = counter.get(i, 0) + 1
    # counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # return [i[0] for i in counter[:k]]

    from collections import Counter
    import heapq
    counter = Counter(nums)
    return heapq.nlargest(k, counter.keys(), key=counter.get)
# print(topKFrequent([1,1,1,2,2,3], 2))

def productExceptSelf(nums):
    # zero_index = [i for i, v in enumerate(nums) if v == 0]
    # if len(zero_index) == len(nums):
    #     return nums
    # else:
    #     total_product = reduce(lambda x, y: x*y, [i for i in nums if i != 0])
    #     if len(zero_index) == 1:
    #         return [total_product if i in zero_index else 0 for i, v in enumerate(nums)]
    #     elif len(zero_index) > 1:
    #         return [0] * len(nums)
    #     else:
    #         return [int(total_product/v) for v in nums]

    # out = []
    # p = 1
    # for i in range(0, len(nums)):
    #     out.append(p)
    #     p *= nums[i]
    # p = 1
    # for i in range(len(nums)-1, -1, -1):
    #     out[i] = out[i] * p
    #     p *= nums[i]
    # return out

    # one pass:
    ans = [1 for _ in nums]
    left, right = 1, 1
    for i in range(len(nums)):
        ans[i] *= left
        ans[~i] *= right
        left *= nums[i]
        right *= nums[~i]
    return ans
# print(productExceptSelf([1,2,3]))

def groupAnagrams(strs):
    str_dict = {}
    for i, v in enumerate(strs):
        tmp = ''.join(sorted(v))
        str_dict[tmp] = str_dict.get(tmp, []) + [v]
    return list(str_dict.values())
# print(groupAnagrams([""]))

def findKthLargest(nums, k):
    # return sorted(nums, reverse=True)[k-1]
    import heapq
    return heapq.nlargest(k, nums)[-1]
# print(findKthLargest([3,2,3,1,2,4,5,5,6], 4))

def findDuplicates(nums):
#     nums.sort()
#     for i in range(len(nums)-1):
#         if nums[i] == nums[i + 1]:
#             return nums[i]
    seen = set()
    for i in nums:
        if i not in seen:
            seen.add(i)
        else:
            return i
# print(findDuplicates([1,3,4,2,2]))

# def kthSmallest(matrix, k):
#     smallest_lst = [i[0] for i in matrix]
#     while k > 1:
#         smallest = min(smallest_lst)
#         smallest_idx = smallest_lst.index(smallest)
#         matrix[smallest_idx].pop(0)
#             if len(matrix[0])

def plusOne(digits):
    # res = str(int(''.join([str(i) for i in digits])) + 1)
    # return [int(i) for i in res]
    i = len(digits)-1
    while i >= 0:
        if digits[i] == 9:
            digits[i] = 0
            i -= 1
        else:
            digits[i] += 1
            break
    if i < 0:
        digits = [1] + digits
    return digits
# print(plusOne([9,9,9]))

def uniquePaths(m, n):
    # def calculate_path(m, n, x, y):
    #     if (m > x and n > y):
    #         return calculate_path(m, n, x + 1, y) + calculate_path(m, n, x, y + 1)
    #     elif m == x or (n == y):
    #         return 1
    # return calculate_path(m, n, 1, 1)
    if m == 1 or n == 1:
        return 1
    else:
        matrix = [[1] * n] + [[1] + [0] * (n-1) for i in range(m-1)]  # first row all = 1, other rows all start with 1
        for i in range(1, m):
            for j in range(1, n):
                matrix[i][j] = matrix[i-1][j] + matrix[i][j-1]
        return matrix[-1][-1]
# print(uniquePaths(3, 7))

def climbStairs(n):
    if n == 1:
        return 1
    a, b = 1, 2
    for i in range(2, n):
        tmp = b
        b += a
        a = tmp
    return b
# print(climbStairs(3))

def maxSubArray(nums):
    hightest, current = nums[0], nums[0]
    for i in nums[1:]:
        current = max(i, current+i)
        if current > hightest:
            hightest = current
    return hightest
# print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))

def sortColors(nums):  # dutch flag sorting algo, however slower than another self-written sol
    red, white, blue = 0, 0, len(nums)-1
    while white <= blue:
        if nums[white] == 0:
            nums[red], nums[white] = nums[white], nums[red]
            red += 1
            white += 1
        elif nums[white] == 1:
            white += 1
        else:
            nums[white], nums[blue] = nums[blue], nums[white]
            blue -= 1
    return nums
# print(sortColors([2,0,2,1,1,0]))

def mySqrt(x):
    if x == 0:
        return 0
    else:
        out_len = (len(str(x)) + 1) // 2
        start = 10 ** (out_len - 1)
        end = 10 ** (out_len) - 1
        while end - start > 1:
            new_start = start + (end - start)//2
            if x >= new_start ** 2 and x <= end ** 2:
                start = new_start
            else:
                end = new_start
            print(start, end)
        return start
# print(mySqrt(100))

def numIslands(grid):
    cnt = 0
    def rm_neighbors(i, j, grid):
        grid[i][j] = "0"
        if i > 0 and grid[i-1][j] == "1":
            rm_neighbors(i-1, j, grid)
        if i < len(grid)-1 and grid[i+1][j] == "1":
            rm_neighbors(i+1, j, grid)
        if j > 0 and grid[i][j-1] == "1":
            rm_neighbors(i, j-1, grid)
        if j < len(grid[0])-1 and grid[i][j+1] == "1":
            rm_neighbors(i, j+1, grid)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "1":
                cnt += 1
                rm_neighbors(i, j, grid)
    return cnt
# print(numIslands([
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]))

def isPowerOfThree(n):
    if n == 1:
        return True
    while n > 3:
        n = n / 3
        if n != int(n):
            break
    if n == 3:
        return True
    else:
        return False
# print(isPowerOfThree(15))

def canComplateCircuit(gas, cost):
    # gas_left = list(map(lambda x, y: x-y, gas, cost))
    # if sum(gas_left) >= 0:
    #     pos_idx = [i for i, v in enumerate(gas_left) if v >= 0]
    #     for i in pos_idx:
    #         gas_pos = gas_left[i]
    #         j = i + 1
    #         if j == len(gas_left):
    #             j = 0
    #         while j <= len(gas_left):
    #             if j == i:
    #                 return j
    #             gas_pos += gas_left[j]
    #             if gas_pos < 0:
    #                 break
    #             j += 1
    #             if j == len(gas_left):
    #                 j = 0
    #     return -1
    # else:
    #     return -1

    # https://leetcode.com/problems/gas-station/discuss/274646/Python-One-Pass-Greedy
    gas_left = list(map(lambda x, y: x - y, gas, cost))
    if sum(gas_left) >= 0:
        start, pos = 0, 0
        for i in range(len(gas_left)):
            pos += gas_left[i]
            if pos < 0:
                start = i + 1
                pos = 0
                continue
        return start
    else:
        return -1
# print(canComplateCircuit([5], [4]))

def wordBreak(s, wordDict):
    # if (s in wordDict) or (s == ''):
    #     return True
    # else:
    #     s = [s.split(i) for i in wordDict if i in s]
    #     for item in s:
    #         if all(wordBreak(i, wordDict) for i in item):
    #             return True
    #     return False

    yes = [True] + [False] * len(s)
    for i in range(len(s)+1):
        for word in wordDict:
            if len(word) < i + 1:
                print(i, word)
                if yes[i-len(word)] and word == s[i-len(word):i]:
                    yes[i] = True
                    break
    return yes[-1]
# print(wordBreak("leetcode", ["leet", "code"]))

def numSqaures(n):
    if n == 1:
        return 1
    squares = []
    for i in range(1, n):
        squares.append(i ** 2)
        if squares[-1] > n:
            squares.pop(-1)
            break
    cnt, res, tmp = 1, {n}, set()
    while res:
        for i in res:
            for square in squares:
                if i == square:
                    return cnt
                elif i < square:
                    break
                else:
                    tmp.add(i - square)
        res = tmp
        tmp = set()
        cnt += 1
    return cnt
# print(numSqaures(1))


class RandomizedSet:

    def __init__(self):
        self.data = []

    def insert(self, val):
        if val not in self.data:
            self.data.append(val)
            return True
        else:
            return False

    def remove(self, val):
        if val in self.data:
            self.data.remove(val)
        else:
            return False

    def getRandom(self):
        import random
        return random.choice(self.data)

def rob(nums):
    length = len(nums)
    if length < 3:
        return max(nums)
    else:
        nums[2] = nums[0]+ nums[2]
        if length == 3:
            return max(nums)
        else:
            for i in range(3, length):
                nums[i] = max(nums[i-2], nums[i-3]) + nums[i]
            return max(nums)
# print(rob([2,7,9,3,1]))

def searchMatrix(matrix, target):
    # m, n = len(matrix), len(matrix[0])
    # i, j = m//2, n//2
    # def find(i, j):
    #     if i > m -1 or j > n-1 or i < 0 or j < 0:
    #         return False
    #     elif matrix[i][j] == "":
    #         return False
    #     elif target == matrix[i][j]:
    #         return True
    #     elif target < matrix[i][j]:
    #         matrix[i][j] = ""
    #         return find(i-1, j) or find(i, j-1)
    #     else:
    #         matrix[i][j] = ""
    #         return find(i+1, j) or find(i, j+1)
    # return find(i, j)

    m, n = len(matrix), len(matrix[0])
    i, j = 0, n-1
    while i < m and j < n:
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] < target:
            i += 1
        else:
            j -= 1
        if i < 0 or j < 0:
            break
    return False
# print(searchMatrix([[-5]], -10))

# import heapq
# a = []
# for i in [2,3,1,5]:
#     heapq.heappush(a, i)
# print(a)
# heapq.heappop(a)
# print(a)

def countPrimes(n):
    if n < 3:
        return 0
    numbers = list(range(n))
    max_int = n ** 0.5 + 1
    i = 2
    while i < max_int:
        if numbers[i]:
            for j in range(i*i, n, i):
                numbers[j] = 0
        i += 1
    return len(list(filter(lambda x: x >= 2, numbers)))
# print(countPrimes(10))

def lengthOfLIS(nums):
    if len(nums) <= 1:
        return len(nums)
    left = [1]
    for i in range(1, len(nums)):
        largest = 1
        for j in range(i):
            if nums[i] > nums[j]:
                largest = max(largest, 1 + left[j])
        left.append(largest)
    return max(left)
# print(lengthOfLIS([7,7,7,7,7,7,7]))

def reverseVowels(s):
    vowels = ['a', 'e', 'i', 'o', 'u', 'A','E', 'I', 'O', 'U']
    i, j = 0, len(s) - 1
    while i < j:
        if s[i] in vowels and s[j] in vowels:
            print(1)
            s = s[:i] + s[j] + s[i + 1:j] + s[i] + s[j + 1:]
            i += 1
            j -= 1
        elif s[i] in vowels:
            j -= 1
        elif s[j] in vowels:
            i += 1
        else:
            i += 1
            j -= 1
    return s
# print(reverseVowels('Aa'))

def rainwater(S):
    if 'HHH' in S or S.count('-') * 2 < len(S) or S.startswith('HH') or S.endswith('HH'):
        return -1
    for i in range(len(S)):
        if S[i] == 'H':
            if i > 0 and S[i-1] == 'T':
                continue
            else:
                if i < len(S)-1 and S[i+1] == '-':
                    S = S[:i+1] + 'T' + S[i+2:]
                else:
                    S = S[:i-1] + 'T' + S[i:]
    return S.count('T')
# print(rainwater("HH-----"))

def contruct_matrix(U,L,C):
    # 2 * N matrix, U = sum(first row), l = sum(second row), C = 1 * N containing sum of each column. return IMPOSSIBLE if cannot construct
    if U + L != sum(C):
        return "IMPOSSIBLE"
    cnt2 = C.count(2)
    C_str = ''.join([str(i) for i in C])
    if cnt2 > U or cnt2 > L:
        return "IMPOSSIBLE"
    if U == sum(C) or L == sum(C):
        return C_str + ',' + '0' * len(C)
    below_start, upper_remaining, i = 0, U - cnt2, 0
    while upper_remaining > 0:  # can still fill 1 in upper
        if C[i] == 1:
            upper_remaining -= 1
            below_start = i + 1
        i += 1
    front, back = C_str[:below_start], C_str[below_start:]
    upperfront, upperback = front.replace('2', '1'), back.replace('1', '0').replace('2', '1')
    lowerfront = front.replace('1', '0').replace('2', '1')
    lowerback = back.replace('2', '1')
    return upperfront + upperback + ',' + lowerfront + lowerback
# print(contruct_matrix(6,5,[2,0,1, 0, 1,2,0, 1, 2, 0, 1,0,1]))

def find_prime(x, y, p):
    if p > y-x:
        return -1
    tmp = [1 for i in range(y+1)]
    for i in range(2, y+1, 2):
        tmp[i] = 0
    if 2 >= x:
        p -= 1
    for i in range(3, y+1, 2):
        if tmp[i] == 1:
            p -= 1
            if p == 0:
                return i
            else:
                j = 2
                while i * j <= y:
                    tmp[i*j] = 0
                    j += 1
    return -1
# print(find_prime(2, 7, 2))


def get_max_res(lst):
    # max_idx, max_val = 0, 0
    # for i, v in enumerate(lst):
    #     if abs(v) > max_val:
    #         max_idx, max_val = i, abs(v)
    # return sum(lst) - lst[max_idx] + max_val ** 2
    lst.sort()
    print(lst, lst[-1], lst[0])
    if abs(lst[-1]) > abs(lst[0]):
        return sum(lst) - lst[-1] + lst[-1] ** 2
    else:
        return sum(lst) - lst[0] + lst[0] ** 2
print(get_max_res([1,2,4,-3]))
