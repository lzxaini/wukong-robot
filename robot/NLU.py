# -*- coding: utf-8 -*-
from .sdk import unit
from robot import logging, config
from abc import ABCMeta, abstractmethod
import asyncio
from openai import OpenAI
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class AbstractNLU(object):
    """
    Generic parent class for all NLU engines
    """

    __metaclass__ = ABCMeta

    @classmethod
    def get_config(cls):
        return {}

    @classmethod
    def get_instance(cls):
        profile = cls.get_config()
        instance = cls(**profile)
        return instance

    @abstractmethod
    def parse(self, query, **args):
        """
        进行 NLU 解析

        :param query: 用户的指令字符串
        :param **args: 可选的参数
        """
        return None

    @abstractmethod
    def getIntent(self, parsed):
        """
        提取意图

        :param parsed: 解析结果
        :returns: 意图数组
        """
        return None

    @abstractmethod
    def hasIntent(self, parsed, intent):
        """
        判断是否包含某个意图

        :param parsed: 解析结果
        :param intent: 意图的名称
        :returns: True: 包含; False: 不包含
        """
        return False

    @abstractmethod
    def getSlots(self, parsed, intent):
        """
        提取某个意图的所有词槽

        :param parsed: 解析结果
        :param intent: 意图的名称
        :returns: 词槽列表。你可以通过 name 属性筛选词槽，
        再通过 normalized_word 属性取出相应的值
        """
        return None

    @abstractmethod
    def getSlotWords(self, parsed, intent, name):
        """
        找出命中某个词槽的内容

        :param parsed: 解析结果
        :param intent: 意图的名称
        :param name: 词槽名
        :returns: 命中该词槽的值的列表。
        """
        return None

    @abstractmethod
    def getSay(self, parsed, intent):
        """
        提取回复文本

        :param parsed: 解析结果
        :param intent: 意图的名称
        :returns: 回复文本
        """
        return ""


class UnitNLU(AbstractNLU):
    """
    百度UNIT的NLU API.
    """

    SLUG = "unit"

    def __init__(self):
        super(self.__class__, self).__init__()

    @classmethod
    def get_config(cls):
        """
        百度UNIT的配置

        无需配置，所以返回 {}
        """
        return {}

    def parse(self, query, **args):
        """
        使用百度 UNIT 进行 NLU 解析

        :param query: 用户的指令字符串
        :param **args: UNIT 的相关参数
            - service_id: UNIT 的 service_id
            - api_key: UNIT apk_key
            - secret_key: UNIT secret_key
        :returns: UNIT 解析结果。如果解析失败，返回 None
        """
        if (
            "service_id" not in args
            or "api_key" not in args
            or "secret_key" not in args
        ):
            logger.critical(f"{self.SLUG} NLU 失败：参数错误！", stack_info=True)
            return None
        return unit.getUnit(
            query, args["service_id"], args["api_key"], args["secret_key"]
        )

    def getIntent(self, parsed):
        """
        提取意图

        :param parsed: 解析结果
        :returns: 意图数组
        """
        return unit.getIntent(parsed)

    def hasIntent(self, parsed, intent):
        """
        判断是否包含某个意图

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :returns: True: 包含; False: 不包含
        """
        return unit.hasIntent(parsed, intent)

    def getSlots(self, parsed, intent):
        """
        提取某个意图的所有词槽

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :returns: 词槽列表。你可以通过 name 属性筛选词槽，
        再通过 normalized_word 属性取出相应的值
        """
        return unit.getSlots(parsed, intent)

    def getSlotWords(self, parsed, intent, name):
        """
        找出命中某个词槽的内容

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :param name: 词槽名
        :returns: 命中该词槽的值的列表。
        """
        return unit.getSlotWords(parsed, intent, name)

    def getSlotOriginalWords(self, parsed, intent, name):
        """
        找出命中某个词槽的原始内容

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :param name: 词槽名
        :returns: 命中该词槽的值的列表。
        """
        return unit.getSlotOriginalWords(parsed, intent, name)

    def getSay(self, parsed, intent):
        """
        提取 UNIT 的回复文本

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :returns: UNIT 的回复文本
        """
        return unit.getSay(parsed, intent)


class OllamaNLU(AbstractNLU):
    SLUG = "ollama"

    def __init__(self, **kwargs):
        super().__init__()
        
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.model = kwargs.get("model")
        # self.sys_nlu = kwargs.get("sys_nlu")
        self.sys_nlu = self._load_prompt(kwargs.get("sys_nlu"))

        # 校验：缺任何一项都报错，避免“<nil>”错误
        for key in ("base_url", "api_key", "model", "sys_nlu"):
            if getattr(self, key) in (None, ""):
                raise ValueError(f"配置项 `{key}` 缺失，请检查你的 config.yaml")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _load_prompt(self, path_or_text: str, default=""):
        if path_or_text and Path(path_or_text).exists():
            return Path(path_or_text).read_text(encoding="utf-8")
        return path_or_text or default

    @classmethod
    def get_config(cls):

        return config.get("ollama", {})

    async def _request(self, query, sys_nlu=None):
        prompt = sys_nlu or self.sys_nlu
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )
        return completion.choices[0].message.content

    def parse(self, query, **args):
        """
        只解析意图槽位，不返回聊天回复文本
        """
        try:
            # 调用 Ollama，期望返回 JSON 结构意图槽位
            # content = asyncio.run(self._request(query))
            # 调用本地大模型，阻塞式调用
            loop = asyncio.get_event_loop()
            if loop.is_running():
                result = asyncio.run(self._request(query))
            else:
                result = loop.run_until_complete(self._request(query))

            logger.info(f"OllamaNLU 原始返回: {result}")
            # 期望大模型返回标准JSON结构
            try:
                obj = json.loads(result)
                return obj
            except Exception:
                # 如果不是JSON，作为普通聊天
                return {"intent": "chat", "text": result}
        except Exception as e:
            logger.error(f"OllamaNLU 解析失败: {e}", stack_info=True)
            return {"intent": "chat", "text": ""}

    def getIntent(self, parsed):
        if isinstance(parsed, dict) and "intent" in parsed:
            return [parsed["intent"]]
        return []

    def hasIntent(self, parsed, intent):
        return parsed.get("intent") == intent

    def getSlots(self, parsed, intent):
        # 返回所有槽位（去掉 intent 和 text 字段）
        if isinstance(parsed, dict):
            return {k: v for k, v in parsed.items() if k not in ("intent", "text")}
        return {}

    def getSlotWords(self, parsed, intent, name):
        if isinstance(parsed, dict) and name in parsed:
            return [parsed[name]]
        return []


def get_engine_by_slug(slug=None):
    """
    Returns:
        An NLU Engine implementation available on the current platform

    Raises:
        ValueError if no speaker implementation is supported on this platform
    """

    if not slug or type(slug) is not str:
        raise TypeError("无效的 NLU slug '%s'", slug)

    selected_engines = list(
        filter(
            lambda engine: hasattr(engine, "SLUG") and engine.SLUG == slug,
            get_engines(),
        )
    )

    if len(selected_engines) == 0:
        raise ValueError(f"错误：找不到名为 {slug} 的 NLU 引擎")
    else:
        if len(selected_engines) > 1:
            logger.warning(f"注意: 有多个 NLU 名称与指定的引擎名 {slug} 匹配")
        engine = selected_engines[0]
        logger.info(f"使用 {engine.SLUG} NLU 引擎")
        return engine.get_instance()


def get_engines():
    def get_subclasses(cls):
        subclasses = set()
        for subclass in cls.__subclasses__():
            subclasses.add(subclass)
            subclasses.update(get_subclasses(subclass))
        return subclasses

    return [
        engine
        for engine in list(get_subclasses(AbstractNLU))
        if hasattr(engine, "SLUG") and engine.SLUG
    ]
