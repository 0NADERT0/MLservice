from fastapi import FastAPI, UploadFile, Depends
from sqlalchemy import select, insert
from app.models.tables import result
from app.src.db import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from app.src.funcs import save_image, get_result_from_ml, load_ml

app = FastAPI(title="ML-service")

@app.on_event("startup")
def start():
    load_ml()

@app.get("/")
async def main():
    return {"Загрузите изображение автомобиля используя /load-file"}

@app.post("/load-file")
async def upload_file(file: UploadFile, session: AsyncSession = Depends(get_async_session)):
    try:
        contents = await file.read()
        path = save_image(contents, file.filename)
        
        res = get_result_from_ml(path)

        stmt = insert(result).values(file_name=file.filename, file_path=path, predict=res["class_name"])
        await session.execute(stmt)
        await session.commit()

        return res

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
