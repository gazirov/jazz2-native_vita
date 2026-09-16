#include "TileSet.h"

#include <cstring>

#if defined(DEATH_TARGET_VITA)
#	include "../../nCine/Graphics/RHI/GXM/GxmDevice.h"
#endif

namespace Jazz2::Tiles
{
	TileSet::TileSet(StringView path, std::uint16_t tileCount, SmallVector<std::unique_ptr<Texture>, 1>&& textureDiffuse,
		std::unique_ptr<uint8_t[]> mask, std::uint32_t maskSize, std::unique_ptr<Color[]> captionTile, const std::uint8_t* tileDiffuseOpaque
#if defined(DEATH_TARGET_VITA)
		, SmallVector<Array<std::uint8_t>, 1>&& indexedDiffuseTexels
#endif
	)
		: FilePath(path), TextureDiffuse(std::move(textureDiffuse)), _mask(std::move(mask)), _captionTile(std::move(captionTile)),
			_isMaskEmpty(), _isMaskFilled(), _isTileFilled(), _isColumnContiguous()
#if defined(DEATH_TARGET_VITA)
			, _indexedDiffuseTexels(std::move(indexedDiffuseTexels)), _bakedDiffuse()
#endif
	{
		// TilesPerRow/TilesPerTexture are used only for rendering. Every chunk shares the layout of chunk 0
		// (the last one may be shorter), so its size defines how many tiles each chunk covers.
		if (!TextureDiffuse.empty() && TextureDiffuse[0] != nullptr) {
			Vector2i texSize = TextureDiffuse[0]->GetSize();
			TilesPerRow = (texSize.X / (DefaultTileSize + 2));
			TilesPerTexture = TilesPerRow * (texSize.Y / (DefaultTileSize + 2));
		} else {
			TilesPerRow = 0;
			TilesPerTexture = 0;
		}

		TileCount = tileCount;
		_isMaskEmpty.resize(ValueInit, TileCount);
		_isMaskFilled.resize(ValueInit, TileCount);
		_isTileFilled.resize(ValueInit, TileCount);
		_isColumnContiguous.resize(ValueInit, TileCount);
		// 2 bytes per column (first/last solid row); zero-initialized by make_unique
		_columnSpans = std::make_unique<std::uint8_t[]>((std::size_t)TileCount * DefaultTileSize * 2);

		std::uint32_t maskMaxTiles = maskSize / MaskBytesPerTile;

		for (std::uint32_t i = 0; i < tileCount; i++) {
			bool maskEmpty = true;
			bool maskFilled = true;

			if (i < maskMaxTiles) {
				// The mask is packed (1 bit per pixel, the cache file format), so empty/filled collapse
				// to byte compares over the tile's 128 bytes
				auto* maskOffset = &_mask[i * MaskBytesPerTile];
				for (std::int32_t j = 0; j < MaskBytesPerTile; j++) {
					maskEmpty &= (maskOffset[j] == 0x00);
					maskFilled &= (maskOffset[j] == 0xFF);
				}
			}

			if (maskEmpty) {
				_isMaskEmpty.set(i);
			}
			if (maskFilled) {
				_isMaskFilled.set(i);
			}

			// A tile is "filled" for rendering when its diffuse is fully opaque (used to cull hidden debris).
			// The flag is computed from the diffuse alpha by the content loader; it is absent in headless
			// mode, where rendering - and therefore this optimization - does not run.
			if (tileDiffuseOpaque != nullptr && tileDiffuseOpaque[i] != 0) {
				_isTileFilled.set(i);
			}

			// Precompute per-column solid spans. A tile whose every column is vertically contiguous
			// (no solid-empty-solid gaps) can answer "is any pixel in this sub-rectangle solid?" with
			// an exact per-column span overlap test, avoiding the O(width*height) per-pixel scan.
			// Fully filled tiles are handled by an earlier IsTileMaskFilled() early-out, and empty
			// tiles by IsTileMaskEmpty(), so this mainly accelerates slopes and other partial tiles.
			std::uint8_t* spans = &_columnSpans[(std::size_t)i * DefaultTileSize * 2];
			bool columnContiguous = (i < maskMaxTiles && !maskEmpty);
			if (columnContiguous) {
				// Load each packed row once as a 32-bit word, then walk columns as bit tests
				const std::uint8_t* maskOffset = &_mask[i * MaskBytesPerTile];
				std::uint32_t rows[DefaultTileSize];
				for (std::int32_t row = 0; row < DefaultTileSize; row++) {
					rows[row] = GetTileMaskRow(maskOffset, row);
				}
				for (std::int32_t col = 0; col < DefaultTileSize; col++) {
					std::int32_t firstSolid = -1, lastSolid = -1, solidCount = 0;
					for (std::int32_t row = 0; row < DefaultTileSize; row++) {
						if ((rows[row] >> col) & 1) {
							if (firstSolid < 0) {
								firstSolid = row;
							}
							lastSolid = row;
							solidCount++;
						}
					}
					if (firstSolid < 0) {
						spans[col * 2] = 0xFF; // Empty column (0xFF first row never satisfies "<= bottom")
						spans[col * 2 + 1] = 0xFF;
					} else {
						spans[col * 2] = (std::uint8_t)firstSolid;
						spans[col * 2 + 1] = (std::uint8_t)lastSolid;
						if ((lastSolid - firstSolid + 1) != solidCount) {
							// Column has a gap, span would over-report; fall back to per-pixel scan
							columnContiguous = false;
						}
					}
				}
			}
			if (columnContiguous) {
				_isColumnContiguous.set(i);
			}
		}
	}

#if defined(DEATH_TARGET_VITA)
	Texture* TileSet::GetBakedDiffuse(std::int32_t chunk, std::uint16_t paletteOffset, ArrayView<const std::uint32_t> palettes,
		BakedDiffuseFailure* failure)
	{
		if (failure != nullptr) {
			*failure = BakedDiffuseFailure::None;
		}
		if (!IsIndexed || chunk < 0 || chunk >= std::int32_t(TextureDiffuse.size()) || chunk >= std::int32_t(_indexedDiffuseTexels.size()) ||
			TextureDiffuse[chunk] == nullptr || paletteOffset + 256u > palettes.size()) {
			if (failure != nullptr) {
				*failure = BakedDiffuseFailure::InvalidSource;
			}
			return nullptr;
		}

		BakedDiffuse* baked = nullptr;
		for (BakedDiffuse& candidate : _bakedDiffuse) {
			if (candidate.PaletteOffset == paletteOffset) {
				baked = &candidate;
				break;
			}
		}
		if (baked == nullptr) {
			baked = &_bakedDiffuse.emplace_back();
			baked->PaletteOffset = paletteOffset;
		}

		const std::uint32_t* palette = palettes.data() + paletteOffset;
		if (baked->Textures.empty() || std::memcmp(baked->Palette.data(), palette, baked->Palette.size() * sizeof(std::uint32_t)) != 0) {
			baked->Textures.clear();
			std::memcpy(baked->Palette.data(), palette, baked->Palette.size() * sizeof(std::uint32_t));
			for (std::int32_t i = 0; i < std::int32_t(TextureDiffuse.size()); i++) {
				const Vector2i size = TextureDiffuse[i]->GetSize();
				const std::size_t texelCount = std::size_t(size.X) * size.Y;
				const Array<std::uint8_t>& indices = _indexedDiffuseTexels[i];
				if (indices.size() != texelCount) {
					baked->Textures.clear();
					if (failure != nullptr) {
						*failure = BakedDiffuseFailure::MissingIndexedTexels;
					}
					return nullptr;
				}

				Array<std::uint8_t> texels(NoInit, texelCount * 4);
				for (std::size_t texel = 0; texel < texelCount; texel++) {
					const std::uint32_t color = palette[indices[texel]];
					texels[texel * 4 + 0] = std::uint8_t(color >> 0);
					texels[texel * 4 + 1] = std::uint8_t(color >> 8);
					texels[texel * 4 + 2] = std::uint8_t(color >> 16);
					texels[texel * 4 + 3] = std::uint8_t(color >> 24);
				}

				auto texture = std::make_unique<Texture>(FilePath.data(), Texture::Format::RGBA8, size.X, size.Y);
				if (!texture->LoadFromTexels(texels.data(), 0, 0, size.X, size.Y)) {
					baked->Textures.clear();
					if (failure != nullptr) {
						*failure = BakedDiffuseFailure::TextureUpload;
					}
					return nullptr;
				}
				texture->SetMinFiltering(SamplerFilter::Nearest);
				texture->SetMagFiltering(SamplerFilter::Nearest);
				baked->Textures.push_back(std::move(texture));
			}
		}

		return (chunk < std::int32_t(baked->Textures.size()) ? baked->Textures[chunk].get() : nullptr);
	}
#endif

	bool TileSet::OverrideTileDiffuse(std::int32_t tileId, StaticArrayView<(DefaultTileSize + 2) * (DefaultTileSize + 2), std::uint32_t> tileDiffuse)
	{
		if (tileId >= TileCount) {
			return false;
		}

		// The tile may live in any texture chunk when the atlas was split by the device texture-size limit
		const std::int32_t textureChunk = (TilesPerTexture > 0 ? tileId / TilesPerTexture : 0);
		std::int32_t localTileId = tileId;
		Texture* texture = ResolveTextureDiffuse(localTileId);
		if (texture == nullptr) {
			return false;
		}

#if defined(DEATH_TARGET_VITA)
		if (IsIndexed && textureChunk < std::int32_t(_indexedDiffuseTexels.size())) {
			// Level-cache overrides for indexed tiles retain their palette index in red and use alpha only for
			// transparency. Update the retained atlas so the next baked palette variant includes this tile.
			const Vector2i textureSize = texture->GetSize();
			Array<std::uint8_t>& indices = _indexedDiffuseTexels[textureChunk];
			const std::int32_t x = (localTileId % TilesPerRow) * (DefaultTileSize + 2);
			const std::int32_t y = (localTileId / TilesPerRow) * (DefaultTileSize + 2);
			if (indices.size() == std::size_t(textureSize.X) * textureSize.Y) {
				for (std::int32_t row = 0; row < DefaultTileSize + 2; row++) {
					for (std::int32_t column = 0; column < DefaultTileSize + 2; column++) {
						const std::uint32_t color = tileDiffuse[row * (DefaultTileSize + 2) + column];
						indices[(y + row) * textureSize.X + x + column] = ((color >> 24) != 0 ? std::uint8_t(color) : 0);
					}
				}
				_bakedDiffuse.clear();
			}
		}
#endif

		std::int32_t x = (localTileId % TilesPerRow) * (DefaultTileSize + 2);
		std::int32_t y = (localTileId / TilesPerRow) * (DefaultTileSize + 2);

		// The incoming tile is RGBA (palette index in red, alpha in alpha). Repack it to match the atlas format,
		// which may have been reduced to R8 (index only) or RG8 (index + alpha) to save VRAM (see CreateIndexedTexture)
		constexpr std::int32_t Count = (DefaultTileSize + 2) * (DefaultTileSize + 2);
		std::uint32_t channels = texture->GetChannelCount();
		// TODO: _isTileFilled is not properly set
		if (channels == 1) {
			std::uint8_t packed[Count];
			for (std::int32_t i = 0; i < Count; i++) {
				std::uint32_t c = tileDiffuse[i];
				packed[i] = (((c >> 24) & 0xFF) == 0 ? 0 : (std::uint8_t)(c & 0xFF));
			}
			return texture->LoadFromTexels(packed, x, y, DefaultTileSize + 2, DefaultTileSize + 2);
		} else if (channels == 2) {
			std::uint8_t packed[Count * 2];
			for (std::int32_t i = 0; i < Count; i++) {
				std::uint32_t c = tileDiffuse[i];
				packed[(i * 2) + 0] = (std::uint8_t)(c & 0xFF);
				packed[(i * 2) + 1] = (std::uint8_t)((c >> 24) & 0xFF);
			}
			return texture->LoadFromTexels(packed, x, y, DefaultTileSize + 2, DefaultTileSize + 2);
		} else {
			return texture->LoadFromTexels((std::uint8_t*)tileDiffuse.data(), x, y, DefaultTileSize + 2, DefaultTileSize + 2);
		}
	}

	bool TileSet::OverrideTileMask(std::int32_t tileId, StaticArrayView<DefaultTileSize * DefaultTileSize, std::uint8_t> tileMask)
	{
		if (tileId >= TileCount) {
			return false;
		}

		// The level cache delivers overridden masks byte-per-pixel; pack them into the tile's 1-bit store
		auto* maskOffset = &_mask[tileId * MaskBytesPerTile];
		std::memset(maskOffset, 0, MaskBytesPerTile);

		bool maskEmpty = true;
		bool maskFilled = true;
		for (std::int32_t j = 0; j < DefaultTileSize * DefaultTileSize; j++) {
			bool masked = (tileMask[j] > 0);
			if (masked) {
				maskOffset[j >> 3] |= std::uint8_t(1u << (j & 7));
			}
			maskEmpty &= !masked;
			maskFilled &= masked;
		}

		_isMaskEmpty.set(tileId, maskEmpty);
		_isMaskFilled.set(tileId, maskFilled);
		// Disable the column-span fast path for overridden masks, the per-pixel scan stays correct
		_isColumnContiguous.set(tileId, false);

		return true;
	}
}
