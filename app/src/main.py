import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Depends, HTTPException
from sqlalchemy import select, insert
from starlette.responses import StreamingResponse
from app.models.tables import result
from app.src.db import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from app.src.funcs import save_image, get_result_from_ml, load_ml, highlight_damage


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ml()
    yield

app = FastAPI(title="ML-service", lifespan=lifespan)


@app.get("/")
async def main():
    return {"Загрузите изображение автомобиля используя /load-file"}

@app.post("/load-file")
async def upload_file(file: UploadFile, session: AsyncSession = Depends(get_async_session)):
    try:
        contents = await file.read()
        path = save_image(contents, file.filename)
        
        res = get_result_from_ml(path)
        modified_image_bytes = highlight_damage(path, res)

        stmt = insert(result).values(file_name=file.filename, file_path=path, predict=res["class_name"])
        await session.execute(stmt)
        await session.commit()

        return StreamingResponse(io.BytesIO(modified_image_bytes), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))