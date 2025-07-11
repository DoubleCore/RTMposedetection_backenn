"""
自定义异常类
"""

class BaseAnalysisError(Exception):
    """分析系统基础异常"""
    pass

class InputError(BaseAnalysisError):
    """输入处理异常"""
    pass

class ModelError(BaseAnalysisError):
    """模型相关异常"""
    pass

class ProcessingError(BaseAnalysisError):
    """处理流程异常"""
    pass

class ValidationError(BaseAnalysisError):
    """验证异常"""
    pass

class ConfigurationError(BaseAnalysisError):
    """配置异常"""
    pass

class FileManagerError(BaseAnalysisError):
    """文件管理异常"""
    pass 