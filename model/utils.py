import math

def round_width(width, multiplier, min_depth=8, divisor=8):
    """
    Round width of filters based on width multiplier
    from: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py

    Args:
        width (int): the channel dimensions of the input.
        multiplier (float): the multiplication factor.
        min_width (int, optional): the minimum width after multiplication.
            Defaults to 8.
        divisor (int, optional): the new width should be dividable by divisor.
            Defaults to 8.
    """
    if not multiplier:
        return width

    width *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(
        min_depth, int(width + divisor / 2) // divisor * divisor
    )
    if new_filters < 0.9 * width:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
  """
  Round number of layers based on depth multiplier.
  Reference: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py
  """
  multiplier = multiplier
  if not multiplier:
      return repeats
  return int(math.ceil(multiplier * repeats))