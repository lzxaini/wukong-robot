import time

from snowboy import snowboydecoder
from robot import config, logging, utils, constants

logger = logging.getLogger(__name__)

detector = None
recorder = None
porcupine = None


def initDetector(wukong):
    """
    初始化离线唤醒热词监听器，支持 snowboy 和 porcupine 两大引擎
    2025-6-24 lzx 新增了小云小云，首次运行会自动下载离线包
    后续加载大概在15s左右
    """
    global porcupine, recorder, detector
    if config.get("detector", "snowboy") == "porcupine":
        logger.info("使用 porcupine 进行离线唤醒")

        import pvporcupine
        from pvrecorder import PvRecorder

        access_key = config.get("/porcupine/access_key")
        keyword_paths = config.get("/porcupine/keyword_paths")
        keywords = config.get("/porcupine/keywords", ["porcupine"])
        if keyword_paths:
            porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[constants.getConfigData(kw) for kw in keyword_paths],
                sensitivities=[config.get("sensitivity", 0.5)] * len(keyword_paths),
            )
        else:
            porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=keywords,
                sensitivities=[config.get("sensitivity", 0.5)] * len(keywords),
            )

        recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
        recorder.start()

        try:
            while True:
                pcm = recorder.read()

                result = porcupine.process(pcm)
                if result >= 0:
                    kw = keyword_paths[result] if keyword_paths else keywords[result]
                    logger.info(
                        "[porcupine] Keyword {} Detected at time {}".format(
                            kw,
                            time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(time.time())
                            ),
                        )
                    )
                    wukong._detected_callback(False)
                    recorder.stop()
                    wukong.conversation.interrupt()
                    query = wukong.conversation.activeListen()
                    wukong.conversation.doResponse(query)
                    recorder.start()
        except pvporcupine.PorcupineActivationError as e:
            logger.error("[Porcupine] AccessKey activation error", stack_info=True)
            raise e
        except pvporcupine.PorcupineActivationLimitError as e:
            logger.error(
                f"[Porcupine] AccessKey {access_key} has reached it's temporary device limit",
                stack_info=True,
            )
            raise e
        except pvporcupine.PorcupineActivationRefusedError as e:
            logger.error(
                "[Porcupine] AccessKey '%s' refused" % access_key, stack_info=True
            )
            raise e
        except pvporcupine.PorcupineActivationThrottledError as e:
            logger.error(
                "[Porcupine] AccessKey '%s' has been throttled" % access_key,
                stack_info=True,
            )
            raise e
        except pvporcupine.PorcupineError as e:
            logger.error("[Porcupine] 初始化 Porcupine 失败", stack_info=True)
            raise e
        except KeyboardInterrupt:
            logger.info("Stopping ...")
        finally:
            porcupine and porcupine.delete()
            recorder and recorder.delete()

    elif config.get("detector", "snowboy") == "funAsrXiaoYun":
        logger.info("使用 funasr 进行离线关键词唤醒")
        from funasr import AutoModel
        import sounddevice as sd
        import numpy as np
        import queue
        import threading

        # 初始化模型
        funasr_model = AutoModel(
            model="iic/speech_sanm_kws_phone-xiaoyun-commands-offline",
            keywords="小云小云",
            output_dir="./outputs/debug",
            device="cpu",
            disable_update=True
        )

        SAMPLE_RATE = 16000
        CHUNK_DURATION = 1  # 每块 1 秒
        CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"音频输入状态异常: {status}")
            audio_queue.put(indata.copy())

        # 控制检测线程的事件
        detecting_event = threading.Event()
        detecting_event.set()  # 初始允许检测
        def detection_loop():
            logger.info("🎙️ 正在监听唤醒词“小云小云”...")
            while True:
                detecting_event.wait()  # 如果被清除则暂停检测
                audio_chunk = audio_queue.get()
                audio_data = audio_chunk.flatten().astype(np.float32)
                try:
                    result = funasr_model.generate(input=audio_data, input_fs=SAMPLE_RATE)
                    if result and isinstance(result, list):
                        for item in result:
                            text = item.get("text", "")
                            if text.startswith("detected") and "小云小云" in text:
                                # 提取置信度
                                try:
                                    confidence = float(text.strip().split()[-1])
                                except Exception:
                                    confidence = 1.0  # 解析失败时默认可信
                                threshold = config.get("sensitivity", 0.5)
                                if confidence >= threshold:
                                    logger.info(f"✅ 检测到唤醒词: {text} (可信度: {confidence} ≥ 阈值: {threshold})")
                                    # 1. 停止检测
                                    detecting_event.clear()
                                    stream.stop()
                                    # 2. 唤醒流程
                                    wukong.conversation.interrupt()  # <--- 新增，保证和snowboy一致
                                    query = wukong.conversation.activeListen()
                                    wukong.conversation.doResponse(query)
                                    logger.info("🎙️ 回到唤醒监听中...")
                                    # 3. 恢复检测
                                    stream.start()
                                    detecting_event.set()
                                else:
                                    logger.info(f"⚠️ 唤醒词可信度不足: {confidence} < 阈值: {threshold}")
                            else:
                                logger.debug(f"未检测到唤醒词: {text}")
                except Exception as e:
                    logger.error(f"FunASR 检测失败: {e}")

        stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype='float32',
            callback=audio_callback
        )
        stream.start()

        threading.Thread(target=detection_loop, daemon=True).start()

        # 保持主线程不退出
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("主线程退出，停止唤醒监听。")
            stream.stop()
            stream.close()

    else:
        logger.info("使用 snowboy 进行离线唤醒")
        detector and detector.terminate()
        models = constants.getHotwordModel(config.get("hotword", "wukong.pmdl"))
        detector = snowboydecoder.HotwordDetector(
            models, sensitivity=config.get("sensitivity", 0.5)
        )
        # main loop
        try:
            callbacks = wukong._detected_callback
            detector.start(
                detected_callback=callbacks,
                audio_recorder_callback=wukong.conversation.converse,
                interrupt_check=wukong._interrupt_callback,
                silent_count_threshold=config.get("silent_threshold", 15),
                recording_timeout=config.get("recording_timeout", 5) * 4,
                sleep_time=0.03,
            )
            detector.terminate()
        except Exception as e:
            logger.critical(f"离线唤醒机制初始化失败：{e}", stack_info=True)
