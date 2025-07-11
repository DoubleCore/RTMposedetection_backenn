"""
跳水运动员姿态分析系统主入口文件
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.routes import router
from config.settings import settings

app = FastAPI(
    title="跳水运动员姿态分析系统",
    description="基于RTMPose和SAM的跳水运动员姿态分析后端服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "跳水运动员姿态分析系统正在运行", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "diving_pose_analysis"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    ) 