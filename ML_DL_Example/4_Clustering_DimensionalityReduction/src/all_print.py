def all_print(var, global_names):
    print(f"---------- {global_names} ----------")
    print(global_names)
    var_name = [name for name in global_names if global_names[name] is var]
    var_name = var_name[0]
    print(f"========== {var_name} ==========")
    print(f"{var_name}_type : ", type(var))
    print(f"{var_name}_len : ", len(var))
    print(f"{var_name}_shape : ", var.shape)
    print(f"{var_name} : ", var)