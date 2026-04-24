import io
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="数据分析平台", page_icon="📊", layout="wide")


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("仅支持 CSV / Excel 文件")


def _basic_overview(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总行数", len(df))
    c2.metric("总列数", len(df.columns))
    c3.metric("缺失值总数", int(df.isna().sum().sum()))
    numeric_cols = df.select_dtypes(include="number").columns
    c4.metric("数值列数量", len(numeric_cols))

    with st.expander("数据类型与缺失值概览", expanded=True):
        profile = pd.DataFrame(
            {
                "列名": df.columns,
                "数据类型": [str(t) for t in df.dtypes],
                "缺失值个数": [int(df[col].isna().sum()) for col in df.columns],
                "唯一值个数": [int(df[col].nunique(dropna=True)) for col in df.columns],
            }
        )
        st.dataframe(profile, use_container_width=True)


def _to_datetime_if_possible(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    converted = pd.to_datetime(series, errors="coerce")
    if converted.notna().sum() >= max(3, len(series) // 3):
        return converted
    return series


def _line_plot(df: pd.DataFrame) -> None:
    st.subheader("📈 可选参数折线可视化")
    all_cols = list(df.columns)
    numeric_cols = list(df.select_dtypes(include="number").columns)

    if not all_cols or not numeric_cols:
        st.info("当前数据不包含可用于绘图的列（需要至少一个数值列）。")
        return

    col1, col2, col3 = st.columns([2, 3, 2])
    x_col = col1.selectbox("X 轴（通常为时间）", options=all_cols, index=0)
    y_cols = col2.multiselect(
        "Y 轴参数（可多选）",
        options=numeric_cols,
        default=numeric_cols[: min(3, len(numeric_cols))],
    )
    color_col: Optional[str] = col3.selectbox(
        "分组着色（可选）",
        options=["<无>"] + all_cols,
        index=0,
    )
    color_col = None if color_col == "<无>" else color_col

    if not y_cols:
        st.warning("请至少选择一个 Y 轴参数。")
        return

    chart_df = df.copy()
    chart_df[x_col] = _to_datetime_if_possible(chart_df[x_col])

    melted = chart_df.melt(
        id_vars=[x_col] + ([color_col] if color_col else []),
        value_vars=y_cols,
        var_name="参数",
        value_name="值",
    )

    line_kwargs = {
        "data_frame": melted,
        "x": x_col,
        "y": "值",
        "color": "参数",
        "markers": True,
        "template": "plotly_white",
    }
    if color_col:
        line_kwargs["line_dash"] = color_col

    fig = px.line(**line_kwargs)
    fig.update_layout(height=520, legend_title_text="参数")
    st.plotly_chart(fig, use_container_width=True)


def _editing_panel(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🛠️ 数据编辑区")

    st.markdown("**1) 列名修改**")
    with st.expander("打开列名批量重命名", expanded=False):
        rename_map = {}
        cols = st.columns(2)
        for idx, col_name in enumerate(df.columns):
            new_name = cols[idx % 2].text_input(f"{col_name} →", value=col_name, key=f"rename_{col_name}")
            if new_name != col_name and new_name.strip():
                rename_map[col_name] = new_name.strip()
        if rename_map and st.button("应用列名修改", type="primary"):
            df = df.rename(columns=rename_map)
            st.success("列名修改已应用。")

    st.markdown("**2) 行列批量删减**")
    with st.expander("打开行列删除工具", expanded=False):
        row_indices = st.text_input("要删除的行号（逗号分隔，基于当前表格索引）", placeholder="例如: 1,3,5")
        drop_cols = st.multiselect("要删除的列", options=list(df.columns))

        if st.button("执行批量删除"):
            if row_indices.strip():
                try:
                    drop_idx = [int(i.strip()) for i in row_indices.split(",") if i.strip()]
                    existing_idx = [i for i in drop_idx if i in df.index]
                    df = df.drop(index=existing_idx)
                except ValueError:
                    st.error("行号格式有误，请输入整数并用逗号分隔。")
            if drop_cols:
                df = df.drop(columns=drop_cols)
            st.success("批量删除已执行。")

    st.markdown("**3) 表格直接编辑（可批量粘贴修改）**")
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
        hide_index=False,
    )

    return edited_df


def _download_area(df: pd.DataFrame) -> None:
    st.subheader("💾 导出数据")
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载为 CSV", data=csv_data, file_name="edited_data.csv", mime="text/csv")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    st.download_button(
        "下载为 Excel",
        data=excel_buffer.getvalue(),
        file_name="edited_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    st.title("📊 数据分析平台")
    st.caption("上传 Excel / CSV 后，可进行概览、可视化、编辑与评论记录。")

    uploaded = st.file_uploader("上传数据文件", type=["csv", "xlsx", "xls"])

    if not uploaded:
        st.info("请先上传一个 CSV 或 Excel 文件开始分析。")
        return

    try:
        df = _read_uploaded_file(uploaded)
    except Exception as exc:
        st.error(f"文件读取失败：{exc}")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["数据概览", "图表分析", "数据编辑", "评论区"])

    with tab1:
        _basic_overview(df)
        st.subheader("原始数据预览")
        st.dataframe(df.head(200), use_container_width=True)

    with tab2:
        _line_plot(df)

    with tab3:
        edited_df = _editing_panel(df.copy())
        _download_area(edited_df)

    with tab4:
        st.subheader("💬 图表评论")
        st.write("可针对当前可视化结果记录你的观察、结论与下一步动作。")
        if "comments" not in st.session_state:
            st.session_state.comments = []

        comment = st.text_area("输入评论", placeholder="例如：3月后温度参数波动显著增大，需要检查采样频率。")
        col_a, col_b = st.columns([1, 5])
        if col_a.button("提交评论", type="primary") and comment.strip():
            st.session_state.comments.append(comment.strip())
            st.success("评论已保存到当前会话。")

        if st.session_state.comments:
            st.markdown("**历史评论**")
            for i, c in enumerate(st.session_state.comments, start=1):
                st.markdown(f"{i}. {c}")


if __name__ == "__main__":
    main()
