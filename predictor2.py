import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# 设置页面配置
st.set_page_config(page_title="心脏病预测系统", layout="wide")


# 加载模型和数据
@st.cache_resource
def load_model():
    return joblib.load('RF.pkl')


@st.cache_data
def load_data():
    return pd.read_csv('X_test.csv')


# 加载模型和数据
model = load_model()
X_test = load_data()

# 特征名称
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# 标题
st.title('心脏病预测系统')
st.markdown('---')

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    st.subheader('基本信息')
    age = st.number_input('年龄（岁）：', min_value=0, max_value=120, value=41)
    sex = st.selectbox('性别：', options=[0, 1], format_func=lambda x: '男' if x == 1 else '女')
    cp = st.selectbox('胸痛类型（cp）：', options=[0, 1, 2, 3])
    trestbps = st.number_input('静息血压（trestbps）：', min_value=50, max_value=200, value=120)
    chol = st.number_input('胆固醇（chol）：', min_value=100, max_value=600, value=157)
    fbs = st.selectbox('空腹血糖>120 mg/dl (fbs)：', options=[0, 1], format_func=lambda x: '是' if x == 1 else '否')

with col2:
    st.subheader('检查指标')
    restecg = st.selectbox('静息心电图（restecg）：', options=[0, 1, 2])
    thalach = st.number_input('最大心率（thalach）：', min_value=60, max_value=220, value=182)
    exang = st.selectbox('运动引发的心绞痛（exang）：', options=[0, 1], format_func=lambda x: '是' if x == 1 else '否')
    oldpeak = st.number_input('运动引起的ST段抑制（oldpeak）：', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox('运动峰值ST段的坡度（slope）：', options=[0, 1, 2])
    ca = st.selectbox('主要血管数量（ca）：', options=[0, 1, 2, 3, 4])
    thal = st.selectbox('地中海贫血（thal）：', options=[0, 1, 2, 3])

# 收集特征值
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
features = np.array([feature_values])

# 预测按钮
if st.button('🔍 开始预测', type='primary'):
    st.markdown('---')

    # 预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.subheader('📊 预测结果')

    # 创建指标显示
    col_result1, col_result2, col_result3 = st.columns(3)

    with col_result1:
        st.metric(
            label="预测结果",
            value="患病" if predicted_class == 1 else "健康",
            delta="高风险" if predicted_class == 1 else "低风险"
        )

    with col_result2:
        if predicted_class == 1:
            st.metric(label="患病概率", value=f"{predicted_proba[1] * 100:.1f}%")
        else:
            st.metric(label="健康概率", value=f"{predicted_proba[0] * 100:.1f}%")

    with col_result3:
        st.metric(label="置信度", value=f"{max(predicted_proba) * 100:.1f}%")

    # 建议信息
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"⚠️ **高风险提醒**\n\n"
            f"根据模型预测，您患心脏病的风险较高。\n\n"
            f"预测患病概率为 **{probability:.1f}%**。\n\n"
            f"📌 **建议**：请立即咨询专业医生，进行进一步检查和评估。\n\n"
            f"💡 **健康建议**：保持健康饮食、规律运动、控制血压和胆固醇。"
        )
    else:
        advice = (
            f"✅ **低风险评估**\n\n"
            f"根据模型预测，您患心脏病的风险较低。\n\n"
            f"预测健康概率为 **{probability:.1f}%**。\n\n"
            f"📌 **建议**：继续保持健康的生活方式，定期进行体检。\n\n"
            f"💡 **健康建议**：保持均衡饮食、适度运动、戒烟限酒、保持良好心态。"
        )

    st.info(advice)

    # SHAP解释
    st.markdown('---')
    st.subheader('🔬 SHAP 模型解释')

    try:
        # SHAP解释
        explainer_shap = shap.TreeExplainer(model)

        # 创建特征DataFrame
        feature_df = pd.DataFrame([feature_values], columns=feature_names)

        # 计算SHAP值
        shap_values = explainer_shap.shap_values(feature_df)

        # 创建SHAP force plot
        fig, ax = plt.subplots(figsize=(12, 3))

        if predicted_class == 1:
            shap.force_plot(
                explainer_shap.expected_value[1],
                shap_values[:, :, 1],
                feature_df,
                matplotlib=True,
                show=False
            )
        else:
            shap.force_plot(
                explainer_shap.expected_value[0],
                shap_values[:, :, 0],
                feature_df,
                matplotlib=True,
                show=False
            )

        plt.tight_layout()
        plt.savefig('shap_force_plot.png', bbox_inches='tight', dpi=300)
        plt.close()

        st.image('shap_force_plot.png', caption='SHAP Force Plot - 各特征对预测的影响')

    except Exception as e:
        st.warning(f"SHAP可视化生成失败: {e}")
        st.info("尝试使用条形图替代...")

        # 备用方案：SHAP条形图
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            if len(shap_values.shape) == 3:
                shap.summary_plot(shap_values[:, :, predicted_class], feature_df,
                                  feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, feature_df,
                                  feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig('shap_bar_plot.png', bbox_inches='tight', dpi=300)
            plt.close()
            st.image('shap_bar_plot.png', caption='SHAP特征重要性')
        except:
            st.error("无法生成SHAP可视化")

    # LIME解释
    st.markdown('---')
    st.subheader('🎯 LIME 局部解释')

    try:
        # 创建LIME解释器
        lime_explainer = LimeTabularExplainer(
            training_data=X_test.values,
            feature_names=X_test.columns.tolist(),
            class_names=['健康', '患病'],
            mode='classification',
            random_state=42
        )

        # 生成LIME解释
        lime_exp = lime_explainer.explain_instance(
            data_row=features.flatten(),
            predict_fn=model.predict_proba
        )

        # 显示LIME结果
        lime_html = lime_exp.as_html(show_table=False)
        st.components.v1.html(lime_html, height=600, scrolling=True)

        # 显示特征贡献
        st.markdown("**特征贡献分析：**")
        lime_list = lime_exp.as_list()
        for feature, weight in lime_list[:5]:
            if weight > 0:
                st.write(f"✅ {feature}: +{weight:.3f} (增加风险)")
            else:
                st.write(f"❌ {feature}: {weight:.3f} (降低风险)")

    except Exception as e:
        st.error(f"LIME解释生成失败: {e}")
        st.info("请确保X_test.csv文件存在且格式正确")

# 侧边栏信息
with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    ### 特征说明
    - **年龄**: 患者的年龄（岁）
    - **性别**: 0=女，1=男
    - **胸痛类型**: 0-3，数值越高疼痛越严重
    - **静息血压**: 静息状态下的血压值
    - **胆固醇**: 血清胆固醇含量
    - **空腹血糖**: 是否大于120 mg/dl
    - **静息心电图**: 心电图结果（0-2）
    - **最大心率**: 运动时达到的最大心率
    - **运动心绞痛**: 运动是否引发心绞痛
    - **ST段抑制**: 运动引起的ST段抑制程度
    - **ST段坡度**: 运动峰值ST段的坡度
    - **主要血管数量**: 0-4条
    - **地中海贫血**: 0-3型

    ### 注意事项
    ⚠️ 本系统仅供学习和参考，不能替代专业医疗诊断
    """)

    st.markdown("---")
    st.caption("© 2024 心脏病预测系统 | 版本 1.0")