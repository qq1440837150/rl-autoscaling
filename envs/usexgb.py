import joblib
import xgboost as xgb

predictor = xgb.XGBRegressor()

predictor = joblib.load("D:\\python\\gym\\gym-hpa\\maxthreads-model.pkl")


# print(predictor.predict([2000]))
def updateModel(x, y):
    # 新数据
    X_new = [[int(x)]]
    y_new = [int(y)]
    booster = predictor.get_booster()
    # 创建 DMatrix 对象
    dtrain_new = xgb.DMatrix(X_new, label=y_new)

    # 获取模型的参数
    params = booster.attributes()

    # 将参数从字符串格式转换为字典
    params = {k: eval(v) for k, v in params.items()}
    params['learning_rate'] = 0.01  # 修改学习率
    params['monotonicity_constraints'] = '(1)'  # 添加单调性约束，假设两个特征



    # 继续训练模型
    booster = xgb.train(params, dtrain_new, num_boost_round=10, xgb_model=booster)

    # 将 booster 保存回 XGBRegressor
    predictor._Booster = booster


def updateModel2():
    # 新数据
    X_new = [[100],[200],[300],[400],[500],[600],[700],[800],[900],[1000],[1200],[1300],[1400],[1500],[1600]]
    y_new = [[1],[1],[2],[2],[4],[4],[6],[6],[8],[8],[10],[10],[12],[16],[18]]
    booster = predictor.get_booster()
    # 创建 DMatrix 对象
    dtrain_new = xgb.DMatrix(X_new, label=y_new)

    # 获取模型的参数
    params = booster.attributes()

    # 将参数从字符串格式转换为字典
    params = {k: eval(v) for k, v in params.items()}
    params['learning_rate'] = 0.01  # 修改学习率
    params['monotonicity_constraints'] = '(1)'  # 添加单调性约束，假设两个特征



    # 继续训练模型
    booster = xgb.train(params, dtrain_new, num_boost_round=10, xgb_model=booster)

    # 将 booster 保存回 XGBRegressor
    predictor._Booster = booster

def saveModel():
    joblib.dump(predictor, "D:\\python\\gym\\gym-hpa\\maxthreads-model.pkl")

if __name__ == '__main__':
    # updateModel(1000,2)
    for i in range(100,10000,100):
        print(str(i)+"---"+str(predictor.predict([i])))
    # saveModel()
    # updateModel(100, 1)
    # updateModel(200, 1)
    # updateModel(300, 2)
    # updateModel(400, 2)
    #
    # updateModel(500,4)
    # updateModel2()
    # updateModel2()
    # updateModel2()
    # updateModel2()
    # updateModel2()
    # updateModel2()
    # updateModel2()
    # updateModel2()

    # saveModel()
