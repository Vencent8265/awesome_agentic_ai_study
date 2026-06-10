import yaml

#1.读取yaml文件
with open("Stage_0/src/config.yaml","r") as f:
    config = yaml.safe_load(f)

print("修改前：",config["app"]["version"])
#2.修改一个值
config["app"]["verison"] = "1.1.0"

#3.写回文件
with open("Stage_0/src/config.yaml","w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("修改后：",config["app"]["verison"])    
print("已写回文件")