from keras import callbacks

# 需要将call_back_list传入model.fit的callbacks参数中
callbacks_list = [
    # 如果精度在多于一轮的时间（即两轮）内不再改善，中断训练
    callbacks.EarlyStopping(
        monitor='acc',
        patience=1,
    ),
    # 只有当val_loss更好时，更新模型参数
    callbacks.ModelCheckpoint(
        filepath='my_model.h5',
        monitor='val_loss',
        save_best_only=True,
    ),
    # 验证损失不再改善loss plateau -> 增大或减小学习率
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
    )
]
