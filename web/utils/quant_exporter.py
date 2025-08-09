#!/usr/bin/env python3
import streamlit as st
from typing import Dict, Any

def render_quant_export_buttons(quant_report: Dict[str, Any]):
    if not isinstance(quant_report, dict):
        return
    st.markdown("---")
    st.caption("量化结果导出")
    # 导出因子样本
    fac = quant_report.get('factors')
    if fac is not None and hasattr(fac, 'to_csv'):
        st.download_button(
            label="下载因子样本CSV",
            data=fac.to_csv(index=False).encode('utf-8'),
            file_name="factors_sample.csv",
            mime="text/csv"
        )
    # 导出简易回测
    bt = quant_report.get('backtest')
    if isinstance(bt, dict):
        import pandas as pd
        st.download_button(
            label="下载简易回测结果CSV",
            data=pd.DataFrame([bt]).to_csv(index=False).encode('utf-8'),
            file_name="backtest_simple.csv",
            mime="text/csv"
        )
    # 导出滚动横截面曲线与摘要
    csr = quant_report.get('cross_section_rolling')
    if isinstance(csr, dict):
        import pandas as pd
        curve = csr.get('cum_curve') or []
        if curve:
            st.download_button(
                label="下载滚动横截面曲线CSV",
                data=pd.DataFrame(curve).to_csv(index=False).encode('utf-8'),
                file_name="cs_rolling_curve.csv",
                mime="text/csv"
            )
        summary = csr.get('summary') or {}
        if summary:
            st.download_button(
                label="下载滚动横截面摘要CSV",
                data=pd.DataFrame([summary]).to_csv(index=False).encode('utf-8'),
                file_name="cs_rolling_summary.csv",
                mime="text/csv"
            )