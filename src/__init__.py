import os

if os.environ.get("TYPECHECK") == "1":
    import tensorflow as tf
    from jaxtyping import install_import_hook

    tf.get_logger().setLevel("ERROR")
    install_import_hook("src", "beartype.beartype")
