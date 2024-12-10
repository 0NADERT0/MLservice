import asyncio
import base64
import io
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi import FastAPI, UploadFile, Depends, HTTPException, Request, Response
from sqlalchemy import select, insert
from starlette.responses import StreamingResponse
from app.models.tables import result
from app.src.db import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from app.src.funcs import save_image, get_result_from_ml, load_ml, highlight_damage, save_predict_image

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ml()
    yield

app = FastAPI(title="ML-service", lifespan=lifespan)

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        start_time = asyncio.get_event_loop().time()
        response = await asyncio.wait_for(call_next(request), timeout=10)
        process_time = asyncio.get_event_loop().time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except asyncio.TimeoutError:
        process_time = asyncio.get_event_loop().time() - start_time
        return JSONResponse({'detail': 'Request processing time exceeded limit', 'processing_time': process_time}, status_code=504)

@app.get("/")
async def main():
    return {"Загрузите изображение автомобиля используя /load-file"}

@app.post("/load-file")
async def upload_file(file: list[UploadFile], session: AsyncSession = Depends(get_async_session)):
    try:
        predicted_images = []
        for file in file:
            contents = await file.read()
            path = save_image(contents, file.filename)

            res = get_result_from_ml(path)
            modified_image_bytes = highlight_damage(path, res)
            img_str = base64.b64encode(modified_image_bytes).decode('utf-8')
            predicted_images.append(
                {"filename": file.filename, "base64": img_str}
            )

            predicted_path = save_predict_image(modified_image_bytes, file.filename)

            stmt = insert(result).values(file_name=file.filename, file_path=path, predict=res["class_name"], predict_file_path=predicted_path)
            await session.execute(stmt)
            await session.commit()

        return JSONResponse(content={"images": predicted_images})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{file_name}")
async def get_results(file_name: str, session: AsyncSession = Depends(get_async_session)):
    try:
        stmt = select(result).where(result.c.file_name == file_name)
        res = await session.execute(stmt)
        analysis_result = res.fetchone()

        if not analysis_result:
            raise HTTPException(status_code=404, detail="File not found")

        path = analysis_result.predict_file_path
        image = Image.open(path)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        img_byte = img_byte_arr.getvalue()

        return StreamingResponse(io.BytesIO(img_byte), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def get_files(session: AsyncSession = Depends(get_async_session)):
    try:
        stmt = select(result)
        res = await session.execute(stmt)
        files = res.fetchall()
        file_list = []
        for file in files:
            file_list.append(
                {
                    "file_name": file.file_name,
                    "file_path": file.file_path,
                    "predict": file.predict
                }
            )

        return file_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
