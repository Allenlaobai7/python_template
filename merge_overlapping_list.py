def merge_overlapping_lst(lst):
    # input: lst of sublists. recursively merge all sublists with intersection. input lst will be altered
    out = []
    original_lst = lst.copy() # for comparision with output
    while True:
        current_lst = lst[0]
        if len(lst) == 1:
            out.append(current_lst) # only 1 item left. add to output as a separate record
            lst.remove(current_lst)
        else:
            overlapped = [v for v in lst if set(v).intersection(set(current_lst))] # find all overlapped records
            if len(overlapped) > 1:
                out.append(list(set([i for v in overlapped for i in v]))) # add all merged profiles to output
                for item in overlapped:
                    lst.remove(item)
            else:
                out.append(current_lst) # no overlaps. add to output as a separate record
                lst.remove(current_lst)
        if not lst:
            break
    if original_lst == out: # nothing changes, recursion stops and return current result
        return original_lst
    else:
        return merge_overlapping_lst(out)
