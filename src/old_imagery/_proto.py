"""Runtime-built protobuf message classes for the Keyhole wire formats.

The upstream C# project ships ``protoc``-generated sources rather than the
``.proto`` files themselves.  Both generated files embed the serialized
``FileDescriptorProto`` they were built from, so the schemas were extracted
verbatim into ``_descriptors/*.desc`` and are loaded into a private descriptor
pool here.  That keeps the schema byte-identical to upstream and avoids both a
``protoc`` build step and any hand-reconstruction guesswork.
"""

from __future__ import annotations

from importlib import resources

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_POOL = descriptor_pool.DescriptorPool()


def _load(name: str) -> None:
    data = (resources.files(__package__) / "_descriptors" / f"{name}.desc").read_bytes()
    _POOL.Add(descriptor_pb2.FileDescriptorProto.FromString(data))


for _name in ("dbroot_v2", "quadtreeset"):
    _load(_name)


def _cls(full_name: str):
    return message_factory.GetMessageClass(_POOL.FindMessageTypeByName(full_name))


EncryptedDbRootProto = _cls("keyhole.dbroot.EncryptedDbRootProto")
DbRootProto = _cls("keyhole.dbroot.DbRootProto")
QuadtreePacket = _cls("keyhole.QuadtreePacket")
QuadtreeNode = _cls("keyhole.QuadtreeNode")
QuadtreeLayer = _cls("keyhole.QuadtreeLayer")

# keyhole.QuadtreeLayer.LayerType
LAYER_TYPE_IMAGERY = 0
LAYER_TYPE_TERRAIN = 1
LAYER_TYPE_VECTOR = 2
LAYER_TYPE_IMAGERY_HISTORY = 3

__all__ = [
    "EncryptedDbRootProto",
    "DbRootProto",
    "QuadtreePacket",
    "QuadtreeNode",
    "QuadtreeLayer",
    "LAYER_TYPE_IMAGERY",
    "LAYER_TYPE_TERRAIN",
    "LAYER_TYPE_VECTOR",
    "LAYER_TYPE_IMAGERY_HISTORY",
]
