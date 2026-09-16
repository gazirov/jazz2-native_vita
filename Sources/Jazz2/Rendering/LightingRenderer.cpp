#include "LightingRenderer.h"
#include "PlayerViewport.h"

#include "../../nCine/Graphics/RenderBuffersManager.h"
#include "../../nCine/Graphics/RenderQueue.h"
#include "../../nCine/Graphics/RenderResources.h"

#include <algorithm>
#include <cmath>

namespace Jazz2::Rendering
{
#if defined(RHI_CAP_SHADERS) && defined(RHI_CAP_FRAMEBUFFERS)
	namespace
	{
		// Interleaved per-vertex format shared with the tile-layer meshes (position.xy, texcoords.xy, color.rgba),
		// so the mesh shaders declare the same attributes - see ContentResolver::CompileShaders()
		constexpr std::uint32_t FloatsPerVertex = 8;
	}
#endif

	LightingRenderer::LightingRenderer(PlayerViewport* owner)
		: _owner(owner)
	{
		_emittedLightsCache.reserve(32);
		setVisitOrderState(SceneNode::VisitOrderState::Disabled);
	}

	bool LightingRenderer::OnDraw(RenderQueue& renderQueue)
	{
#if !defined(RHI_CAP_SHADERS) || !defined(RHI_CAP_FRAMEBUFFERS)
		// Dynamic lighting is composited by a full-screen shader pass into an off-screen buffer, which the
		// combine shader later applies; on backends without cheap programmable shaders (the software renderer)
		// there is no combine pass, so the scene is drawn without dynamic lighting
		return true;
#else
		_emittedLightsCache.clear();

		// Collect all active light emitters
		auto actors = _owner->_levelHandler->GetActors();
		std::size_t actorsCount = actors.size();
		for (std::size_t i = 0; i < actorsCount; i++) {
			actors[i]->OnEmitLights(_emittedLightsCache);
		}
		// Every actor in the level emits, wherever it is, and in splitscreen each viewport collects the same set,
		// so a light whose circle cannot reach this view is dropped before it costs any geometry. Nothing else
		// culls them: the render queue only culls drawable nodes, and these are raw commands.
		const Rectf cullingRect = RenderResources::GetCurrentViewport()->GetCullingRect();
		const float cullMinX = cullingRect.X, cullMaxX = cullingRect.X + cullingRect.W;
		const float cullMinY = cullingRect.Y, cullMaxY = cullingRect.Y + cullingRect.H;

		_vertices.clear();
#if defined(DEATH_TARGET_VITA) && defined(WITH_RHI_GXM)
		// A dense chain of animated fire objects can otherwise submit hundreds of overlapping additive quads.
		// Keep the brightest nearby lights, merge neighbours, then enforce a fixed per-viewport GPU budget.
		constexpr std::uint32_t MaxLightsForMerge = 128;
		constexpr std::uint32_t MaxVisibleLights = 64;
		struct RankedLight {
			LightEmitter Light;
			float Score;
		};
		SmallVector<RankedLight, 0> candidates;
		const float centerX = (cullMinX + cullMaxX) * 0.5f;
		const float centerY = (cullMinY + cullMaxY) * 0.5f;
		const float viewRadiusSq = (cullingRect.W * cullingRect.W + cullingRect.H * cullingRect.H) * 0.25f;
		for (auto& light : _emittedLightsCache) {
			// A light with no far radius covers no pixels at all (the quad the shader path built for it was
			// zero-sized), and its normalized near radius would divide by zero
			if (light.RadiusFar <= 0.0f) {
				continue;
			}
			if (light.Pos.X + light.RadiusFar <= cullMinX || light.Pos.X - light.RadiusFar >= cullMaxX ||
				light.Pos.Y + light.RadiusFar <= cullMinY || light.Pos.Y - light.RadiusFar >= cullMaxY) {
				continue;
			}

			const float dx = light.Pos.X - centerX;
			const float dy = light.Pos.Y - centerY;
			const float proximity = 1.0f / (1.0f + (dx * dx + dy * dy) / std::max(viewRadiusSq, 1.0f));
			candidates.push_back({light, std::max(light.Intensity * light.Brightness * light.RadiusFar, 0.0f) * proximity});
		}
		std::sort(candidates.begin(), candidates.end(), [](const RankedLight& a, const RankedLight& b) {
			return (a.Light.IsPlayerLight != b.Light.IsPlayerLight ? a.Light.IsPlayerLight : a.Score > b.Score);
		});
		if (candidates.size() > MaxLightsForMerge) {
			candidates.resize(MaxLightsForMerge);
		}
		SmallVector<LightEmitter, 0> budgetedLights;
		for (const RankedLight& candidate : candidates) {
			bool merged = false;
			for (LightEmitter& existing : budgetedLights) {
				if (existing.IsPlayerLight != candidate.Light.IsPlayerLight) {
					continue;
				}
				const float mergeDistance = (existing.RadiusFar + candidate.Light.RadiusFar) * 0.5f;
				const float dx = existing.Pos.X - candidate.Light.Pos.X;
				const float dy = existing.Pos.Y - candidate.Light.Pos.Y;
				if (dx * dx + dy * dy > mergeDistance * mergeDistance) {
					continue;
				}

				const float existingWeight = std::max(existing.Intensity * existing.Brightness * existing.RadiusFar, 0.01f);
				const float candidateWeight = std::max(candidate.Light.Intensity * candidate.Light.Brightness * candidate.Light.RadiusFar, 0.01f);
				const Vector2f existingPos = existing.Pos;
				existing.Pos = (existingPos * existingWeight + candidate.Light.Pos * candidateWeight) / (existingWeight + candidateWeight);
				// Moving the centre without expanding the radius cuts off the original cubic falloff and makes
				// merged fire lights look like hard-edged patches. Keep both source circles fully covered.
				const float existingDistance = std::sqrt((existingPos.X - existing.Pos.X) * (existingPos.X - existing.Pos.X) +
					(existingPos.Y - existing.Pos.Y) * (existingPos.Y - existing.Pos.Y));
				const float candidateDistance = std::sqrt((candidate.Light.Pos.X - existing.Pos.X) * (candidate.Light.Pos.X - existing.Pos.X) +
					(candidate.Light.Pos.Y - existing.Pos.Y) * (candidate.Light.Pos.Y - existing.Pos.Y));
				existing.Intensity = std::max(existing.Intensity, candidate.Light.Intensity);
				existing.Brightness = std::max(existing.Brightness, candidate.Light.Brightness);
				existing.RadiusNear = std::min(existing.RadiusNear, candidate.Light.RadiusNear);
				existing.RadiusFar = std::max(existing.RadiusFar + existingDistance, candidate.Light.RadiusFar + candidateDistance);
				merged = true;
				break;
			}
			if (!merged) {
				if (budgetedLights.size() < MaxVisibleLights) {
					budgetedLights.push_back(candidate.Light);
				} else {
				}
			}
		}
		for (const LightEmitter& light : budgetedLights) {
			AppendLightQuad(light);
		}
#else
		for (auto& light : _emittedLightsCache) {
			if (light.RadiusFar <= 0.0f || light.Pos.X + light.RadiusFar <= cullMinX || light.Pos.X - light.RadiusFar >= cullMaxX ||
				light.Pos.Y + light.RadiusFar <= cullMinY || light.Pos.Y - light.RadiusFar >= cullMaxY) {
				continue;
			}
			AppendLightQuad(light);
		}
#endif

		if (_vertices.empty()) {
			return true;
		}

		// Cap each indexed light mesh to both shared buffers. Four vertices plus six indices describe a light's two
		// triangles, avoiding the two duplicated vertices the old non-indexed stream submitted for every light.
		const std::uint32_t maxVertexDataSize = RenderResources::GetBuffersManager().Specs(RenderBuffersManager::BufferTypes::Array).maxSize;
		const std::uint32_t maxIndexDataSize = RenderResources::GetBuffersManager().Specs(RenderBuffersManager::BufferTypes::ElementArray).maxSize;
		std::uint32_t maxVerticesPerChunk = std::min(maxVertexDataSize / (FloatsPerVertex * sizeof(float)),
			(maxIndexDataSize / sizeof(std::uint16_t)) * 2 / 3);
		maxVerticesPerChunk = std::min(maxVerticesPerChunk, std::uint32_t(UINT16_MAX - 3));
		maxVerticesPerChunk -= (maxVerticesPerChunk % 4);
		FATAL_ASSERT(maxVerticesPerChunk >= 4);

		const std::uint32_t totalVertices = (std::uint32_t)(_vertices.size() / FloatsPerVertex);
		const std::uint32_t maxIndicesPerChunk = maxVerticesPerChunk / 4 * 6;
		_indices.resize_for_overwrite(maxIndicesPerChunk);
		for (std::uint32_t firstVertex = 0, firstIndex = 0; firstIndex < maxIndicesPerChunk; firstVertex += 4) {
			_indices[firstIndex++] = std::uint16_t(firstVertex);
			_indices[firstIndex++] = std::uint16_t(firstVertex + 1);
			_indices[firstIndex++] = std::uint16_t(firstVertex + 2);
			_indices[firstIndex++] = std::uint16_t(firstVertex);
			_indices[firstIndex++] = std::uint16_t(firstVertex + 2);
			_indices[firstIndex++] = std::uint16_t(firstVertex + 3);
		}
		std::int32_t commandIndex = 0;
		for (std::uint32_t firstVertex = 0; firstVertex < totalVertices; firstVertex += maxVerticesPerChunk) {
			const std::uint32_t count = std::min(maxVerticesPerChunk, totalVertices - firstVertex);

			RenderCommand* command = RentRenderCommand(commandIndex++);

			// Vertex positions are already in world space, so the model matrix is identity. Re-set every frame
			// because committing it also folds in the depth of the command's layer, which is derived from the
			// camera's clip planes - those are rebuilt whenever the viewport is resized.
			command->SetTransformation(Matrix4x4f::Translation(0.0f, 0.0f, 0.0f));

			auto& geometry = command->GetGeometry();
			geometry.SetElementsPerVertex(FloatsPerVertex);
			geometry.SetVertexCount(count);
			geometry.SetHostVertexPointer(_vertices.data() + firstVertex * FloatsPerVertex);
			geometry.SetIndexCount(count / 4 * 6);
			geometry.SetHostIndexPointer(_indices.data());
			geometry.SetDrawParameters(PrimitiveType::Triangles, 0, count);

			renderQueue.AddCommand(command);
		}

		return true;
#endif
	}

#if defined(RHI_CAP_SHADERS) && defined(RHI_CAP_FRAMEBUFFERS)
	void LightingRenderer::AppendLightQuad(const LightEmitter& light)
	{
		// What the single-light shader derived from its instance block, written out per vertex instead: the quad
		// spans the far radius around the light, the corner offset it measures the distance from is carried in the
		// vertex color's .zw (normalized to [-1, 1], exactly as the old vertex stage handed it over), and the
		// normalized near radius rides in the texture coordinates
		const float x0 = light.Pos.X - light.RadiusFar, x1 = light.Pos.X + light.RadiusFar;
		const float y0 = light.Pos.Y - light.RadiusFar, y1 = light.Pos.Y + light.RadiusFar;
		const float radiusNear = light.RadiusNear / light.RadiusFar;

		std::size_t base = _vertices.size();
		// Every float is written below, so the zero-initialization resize() would do first is wasted work.
		_vertices.resize_for_overwrite(base + 4 * FloatsPerVertex);
		float* v = _vertices.data() + base;
		auto put = [&](float px, float py, float cx, float cy) {
			*v++ = px; *v++ = py; *v++ = radiusNear; *v++ = 0.0f;
			*v++ = light.Intensity; *v++ = light.Brightness; *v++ = cx; *v++ = cy;
		};
		// The index buffer forms the same two triangles the old duplicated-vertex stream used.
		put(x0, y0, -1.0f, -1.0f);
		put(x1, y0,  1.0f, -1.0f);
		put(x1, y1,  1.0f,  1.0f);
		put(x0, y1, -1.0f,  1.0f);
	}

	RenderCommand* LightingRenderer::RentRenderCommand(std::int32_t index)
	{
		if (index < (std::int32_t)_renderCommands.size()) {
			return _renderCommands[index].get();
		}

		// A command's material never changes again after this: the shader is fixed and the mesh needs no texture,
		// so per frame only the geometry pointer and the transformation are rewritten
		auto& command = _renderCommands.emplace_back(std::make_unique<RenderCommand>(RenderCommand::Type::Lighting));
		command->GetMaterial().SetShader(_owner->_levelHandler->_lightingMeshShader);
		command->GetMaterial().SetBlendingEnabled(true);
		command->GetMaterial().SetBlendingFactors(BlendingFactor::SrcAlpha, BlendingFactor::One);
		command->GetMaterial().ReserveUniformsDataMemory();

		auto* textureUniform = command->GetMaterial().Uniform(Material::TextureUniformName);
		if (textureUniform != nullptr && textureUniform->GetIntValue(0) != 0) {
			textureUniform->SetIntValue(0); // GL_TEXTURE0
		}
		// Unused by the mesh shader, which takes every light parameter from the vertex stream, but the instance
		// block is shared with the sprite family and a stale tint would be visible if that ever changed
		auto* instanceBlock = command->GetInstanceBlock();
		if (instanceBlock != nullptr) {
			auto* colorUniform = instanceBlock->GetUniform(Material::ColorUniformName);
			if (colorUniform != nullptr) {
				colorUniform->SetFloatValue(1.0f, 1.0f, 1.0f, 1.0f);
			}
		}

		return command.get();
	}
#endif
}
